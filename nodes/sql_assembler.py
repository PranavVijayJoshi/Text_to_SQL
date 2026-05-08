"""
Assembles sub-query SQL fragments into a single WITH (CTE) query.

Key improvement: handles WITH RECURSIVE nodes correctly.
A node flagged as recursive emits a WITH RECURSIVE ... SELECT block.
The assembler hoists the RECURSIVE definition to the top-level WITH clause
so PostgreSQL never sees a nested WITH inside a CTE body (which is illegal).

Also handles reserved keyword CTE name sanitisation and dialect translation.
"""
import logging
import re

from state import AgentState

try:
    import sqlglot
except ImportError:
    sqlglot = None

logger = logging.getLogger(__name__)

_RESERVED = {
    "or", "and", "not", "in", "is", "as", "on", "by", "to",
    "at", "do", "if", "no", "of", "select", "from", "where",
    "join", "left", "right", "inner", "outer", "full", "group",
    "order", "having", "limit", "offset", "union", "except",
    "intersect", "with", "case", "when", "then", "else", "end",
    "null", "true", "false", "table", "index", "view", "set",
}


def _safe_cte_name(name: str) -> str:
    return f"cte_{name}" if name.lower() in _RESERVED else name


def _transpile(sql: str, dialect: str) -> str:
    if sqlglot is None or not dialect or dialect.lower() in ("postgres", "postgresql"):
        return sql
    try:
        result = sqlglot.transpile(sql, read="postgres", write=dialect.lower())
        return result[0] if result else sql
    except Exception as exc:
        logger.warning("dialect transpile failed (%s): %s", dialect, exc)
        return sql


def _extract_recursive_cte(node_id: str, sql: str) -> tuple[str, str]:
    """
    Given a WITH RECURSIVE ... SELECT block, split into:
    - The recursive CTE definition (to hoist into the outer WITH)
    - A simple SELECT referencing the recursive CTE by name

    Returns (recursive_definition, simple_select).
    recursive_definition is the body of the WITH RECURSIVE block
    (everything between WITH RECURSIVE and the final SELECT).
    simple_select is a SELECT * FROM <cte_name> to use as the node's CTE body.
    """
    sql = sql.strip()

    # Match: WITH RECURSIVE cte_name(...) AS ( ... ) SELECT ...
    # We need to find where the final SELECT starts after the CTE definition
    upper = sql.upper()
    with_rec_match = re.match(r"WITH\s+RECURSIVE\s+", sql, re.IGNORECASE)
    if not with_rec_match:
        # Not actually recursive — return as-is
        return "", sql

    # Find the recursive CTE name
    after_with = sql[with_rec_match.end():]
    cte_name_match = re.match(r"(\w+)", after_with)
    if not cte_name_match:
        return "", sql

    recursive_cte_name = cte_name_match.group(1)

    # Find the last SELECT at top level (after all CTE definitions)
    # Strategy: find balanced parentheses to locate the final SELECT
    depth = 0
    i = with_rec_match.end()
    last_select_pos = -1

    while i < len(sql):
        ch = sql[i]
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                # After closing the last CTE, look for the final SELECT
                rest = sql[i+1:].lstrip()
                if rest.upper().startswith("SELECT"):
                    last_select_pos = i + 1 + (len(sql[i+1:]) - len(rest))
                    break
        i += 1

    if last_select_pos == -1:
        # Can't parse structure — return as flat CTE body
        logger.warning("node '%s': could not parse WITH RECURSIVE structure", node_id)
        return "", sql

    # Everything from WITH RECURSIVE up to (not including) final SELECT
    # becomes the recursive definition to hoist
    recursive_definition = sql[:last_select_pos].strip()
    # Remove the leading "WITH RECURSIVE " so we can embed it in the outer WITH RECURSIVE
    recursive_body = recursive_definition[with_rec_match.end():].strip()

    # The node's CTE body becomes a simple select from the recursive CTE
    simple_select = f"SELECT * FROM {recursive_cte_name}"

    return recursive_body, simple_select


def sql_assembler(state: AgentState) -> AgentState:
    sub_queries = state.get("sub_queries", [])

    if not sub_queries:
        return {
            **state,
            "generated_sql": "",
            "is_valid": False,
            "validation_errors": ["No sub-queries to assemble."],
        }

    # Build safe name map
    name_map = {node["id"]: _safe_cte_name(node["id"]) for node in sub_queries}

    cte_parts: list[str] = []
    recursive_definitions: list[str] = []   # hoisted WITH RECURSIVE bodies
    final_sql: str = ""
    has_recursive = False

    for node in sub_queries:
        sql = (node.get("sql") or "").strip().rstrip(";")
        if not sql:
            logger.warning("node '%s' has empty SQL — skipping", node["id"])
            continue

        # Rename any CTE references using safe name map
        for orig, safe in name_map.items():
            if orig != safe:
                sql = re.sub(rf"\b{re.escape(orig)}\b", safe, sql)

        safe_id = name_map[node["id"]]
        is_recursive = node.get("is_recursive", False) or sql.upper().lstrip().startswith("WITH RECURSIVE")

        if node["id"] == "final":
            final_sql = sql
        elif is_recursive:
            has_recursive = True
            recursive_body, simple_select = _extract_recursive_cte(node["id"], sql)
            if recursive_body:
                # Hoist the recursive definition into the top-level WITH RECURSIVE.
                # Do NOT add a separate CTE entry — the recursive CTE is already
                # named and accessible to all subsequent nodes by its own name.
                recursive_definitions.append(recursive_body)
            else:
                # Fallback: treat as a regular CTE body
                cte_parts.append(f"{safe_id} AS (\n  {sql}\n)")
        else:
            cte_parts.append(f"{safe_id} AS (\n  {sql}\n)")

    # Fallback: promote last node to final if no explicit final node
    if not final_sql and sub_queries:
        last = sub_queries[-1]
        final_sql = (last.get("sql") or "").strip().rstrip(";")
        cte_parts = [p for p in cte_parts if not p.startswith(name_map[last["id"]] + " AS")]
        logger.debug("No 'final' node — promoted '%s' as final SELECT", last["id"])

    if not final_sql:
        return {
            **state,
            "generated_sql": "",
            "is_valid": False,
            "validation_errors": ["Assembly failed: no final SELECT produced."],
        }

    # Build the WITH clause
    # If any recursive definitions were hoisted, use WITH RECURSIVE at the top
    if cte_parts:
        if has_recursive and recursive_definitions:
            # Combine recursive definitions + regular CTEs under one WITH RECURSIVE
            all_cte_defs = recursive_definitions + cte_parts
            with_clause = "WITH RECURSIVE\n" + ",\n".join(all_cte_defs)
        else:
            with_clause = "WITH\n" + ",\n".join(cte_parts)
        assembled = with_clause + "\n" + final_sql
    else:
        assembled = final_sql

    dialect = state.get("sql_dialect", "postgres")
    assembled = _transpile(assembled, dialect)

    logger.debug("assembled SQL (%d chars)", len(assembled))

    return {
        **state,
        "generated_sql": assembled,
        "is_valid": True,
        "validation_errors": [],
        "plan_errors": [],
    }
