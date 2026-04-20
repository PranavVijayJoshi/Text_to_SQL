"""
Assembles sub-query SQL fragments into a single WITH (CTE) query.
Handles dialect translation via sqlglot if sql_dialect != "postgres".
No LLM call — fully deterministic.
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
    if name.lower() in _RESERVED:
        return f"cte_{name}"
    return name


def _transpile(sql: str, dialect: str) -> str:
    """Translate from postgres SQL to the target dialect."""
    if sqlglot is None or not dialect or dialect.lower() in ("postgres", "postgresql"):
        return sql
    try:
        result = sqlglot.transpile(sql, read="postgres", write=dialect.lower())
        return result[0] if result else sql
    except Exception as exc:
        logger.warning("dialect transpile failed (%s): %s", dialect, exc)
        return sql

def sql_assembler(state: AgentState) -> AgentState:
    sub_queries = state.get("sub_queries", [])
    if not sub_queries:
        return {**state, "generated_sql": "", "is_valid": False,
                "validation_errors": ["No sub-queries to assemble."]}

    # Build safe name map: original_id -> safe_id
    name_map = {node["id"]: _safe_cte_name(node["id"]) for node in sub_queries}

    cte_parts = []
    final_sql = ""

    for node in sub_queries:
        sql = (node.get("sql") or "").strip().rstrip(";")
        if not sql:
            continue

        # Rename all CTE references inside this node's SQL
        for orig, safe in name_map.items():
            if orig != safe:
                sql = re.sub(rf"\b{re.escape(orig)}\b", safe, sql)

        safe_id = name_map[node["id"]]

        if node["id"] == "final":
            final_sql = sql
        else:
            cte_parts.append(f"{safe_id} AS (\n  {sql}\n)")

    if not final_sql and sub_queries:
        last = sub_queries[-1]
        final_sql = (last.get("sql") or "").strip().rstrip(";")
        cte_parts = [p for p in cte_parts
                     if not p.startswith(name_map[last["id"]] + " AS")]

    if not final_sql:
        return {**state, "generated_sql": "", "is_valid": False,
                "validation_errors": ["Assembly failed: no final SELECT produced."]}

    assembled = (
        "WITH\n" + ",\n".join(cte_parts) + "\n" + final_sql
        if cte_parts else final_sql
    )

    dialect = state.get("sql_dialect", "postgres")
    assembled = _transpile(assembled, dialect)

    return {**state, "generated_sql": assembled, "is_valid": True,
            "validation_errors": [], "plan_errors": []}






# def sql_assembler(state: AgentState) -> AgentState:
#     sub_queries = state.get("sub_queries", [])

#     if not sub_queries:
#         return {
#             **state,
#             "generated_sql": "",
#             "is_valid": False,
#             "validation_errors": ["No sub-queries to assemble."],
#         }

#     # Separate CTE nodes from the final SELECT node
#     cte_parts: list[str] = []
#     final_sql: str = ""

#     for node in sub_queries:
#         sql = (node.get("sql") or "").strip().rstrip(";")
#         if not sql:
#             logger.warning("node '%s' has empty SQL — skipping", node["id"])
#             continue

#         if node["id"] == "final":
#             final_sql = sql
#         else:
#             cte_parts.append(f"{node['id']} AS (\n  {sql}\n)")

#     # Fallback: if no explicit "final" node, promote the last node
#     if not final_sql and sub_queries:
#         last = sub_queries[-1]
#         final_sql = (last.get("sql") or "").strip().rstrip(";")
#         # Drop it from CTEs if it was already added
#         cte_parts = [p for p in cte_parts if not p.startswith(last["id"] + " AS")]
#         logger.debug("No 'final' node found — promoted '%s' as final SELECT", last["id"])

#     if not final_sql:
#         return {
#             **state,
#             "generated_sql": "",
#             "is_valid": False,
#             "validation_errors": ["Assembly failed: no final SELECT produced by sub-query processor."],
#         }

#     assembled = (
#         "WITH\n" + ",\n".join(cte_parts) + "\n" + final_sql
#         if cte_parts
#         else final_sql
#     )

#     # Dialect translation
#     dialect = state.get("sql_dialect", "postgres")
#     assembled = _transpile(assembled, dialect)

#     logger.debug("assembled SQL (%d chars)", len(assembled))

#     return {
#         **state,
#         "generated_sql": assembled,
#         "is_valid": True,
#         "validation_errors": [],
#         "plan_errors": [],
#     }
