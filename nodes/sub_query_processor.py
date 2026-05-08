"""
Processes each sub-query node in the decomposition DAG.

Two-layer contract enforcement:
  1. Pre-generation: CTE dependency columns injected as hard constraints
     in the HumanMessage so the model sees them at generation time.
  2. Post-generation: sqlglot parses the actual SELECT list and compares
     it against declared output_columns. Mismatches trigger a retry with
     explicit diff feedback.
"""
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage
from llm import llm
from state import AgentState

try:
    import sqlglot
    from sqlglot import exp as sqlglot_exp
except ImportError:
    sqlglot = None
    sqlglot_exp = None

logger = logging.getLogger(__name__)
MAX_NODE_RETRIES = 3


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _strip_sql_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:sql)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _is_recursive_node(node: dict) -> bool:
    intent = node.get("intent", "").lower()
    node_id = node.get("id", "").lower()
    return "recursive" in intent or "category_tree" in node_id or "with recursive" in intent


def _schema_for_node(tables: list[str], state: AgentState) -> str:
    schema = state.get("schema", {})
    lines = []
    for table in tables:
        if table not in schema:
            continue
        lines.append(f"Table: {table}")
        for col in schema[table]:
            parts = [f"  {col['name']} {col['type']}"]
            annot = []
            if col.get("primary_key"):
                annot.append("PK")
            if col.get("foreign_key"):
                annot.append(f"FK → {col['foreign_key']}")
            if col.get("sample_values"):
                samples = ", ".join(str(v) for v in col["sample_values"][:5])
                annot.append(f"samples: {samples}")
            if annot:
                parts.append(f"  ({', '.join(annot)})")
            lines.append("".join(parts))
    return "\n".join(lines) if lines else "No direct tables — references CTEs only."


def _extract_select_columns(sql: str) -> list[str]:
    """
    Parse the SQL with sqlglot and extract the output column aliases
    from the outermost SELECT list.
    Returns a list of lowercase alias/column names, or [] on failure.

    Handles:
    - Plain SELECT
    - WITH ... SELECT  (With expression — final SELECT is .this)
    - WITH RECURSIVE ... SELECT
    """
    if sqlglot is None or sqlglot_exp is None or not sql.strip():
        return []
    try:
        stmts = sqlglot.parse(sql.strip(), read="postgres")
        if not stmts:
            return []
        tree = stmts[0]

        # Resolve the outermost SELECT node
        if isinstance(tree, sqlglot_exp.Select):
            select_node = tree
        elif isinstance(tree, sqlglot_exp.With):
            # .this is the final SELECT after the WITH/WITH RECURSIVE block
            select_node = tree.this if isinstance(tree.this, sqlglot_exp.Select) else None
        else:
            # Fallback: walk() yields (node, parent, key) tuples
            select_node = None
            for node, _parent, _key in tree.walk():
                if isinstance(node, sqlglot_exp.Select):
                    select_node = node
                    break

        if select_node is None:
            return []

        cols = []
        for expr in select_node.expressions:
            if isinstance(expr, sqlglot_exp.Alias):
                cols.append(expr.alias.lower())
            elif isinstance(expr, sqlglot_exp.Column):
                cols.append(expr.name.lower())
            elif isinstance(expr, sqlglot_exp.Star):
                cols.append("*")
            else:
                # Functions, expressions without alias — use sql repr trimmed
                cols.append(expr.sql(dialect="postgres").lower().split(".")[-1][:40])
        return cols
    except Exception as exc:
        logger.debug("_extract_select_columns failed: %s", exc)
        return []


def _validate_output_columns(sql: str, declared: list[str]) -> list[str]:
    """
    Compare actual SELECT output against declared output_columns.
    Returns a list of error strings, empty if valid.
    Skips validation if declared is empty or contains '*'.
    """
    if not declared or "*" in declared:
        return []
    actual = _extract_select_columns(sql)
    if not actual:
        return []  # Can't parse — skip, let syntax/DB validate catch it

    declared_set = {c.lower() for c in declared}
    actual_set = {c.lower() for c in actual if c != "*"}

    missing = sorted(declared_set - actual_set)
    extra = sorted(actual_set - declared_set)

    errors = []
    if missing:
        errors.append(
            f"Output column contract violation — declared but missing from SELECT: {missing}. "
            f"Actual SELECT produces: {actual}. "
            f"Add the missing columns to your SELECT with exact aliases: {missing}."
        )
    if extra:
        errors.append(
            f"Output column contract violation — undeclared columns in SELECT: {extra}. "
            f"Expected only: {declared}. Remove or rename undeclared columns."
        )
    return errors


def _validate_syntax(sql: str) -> list[str]:
    if sqlglot is None or not sql.strip():
        return []
    try:
        stmts = sqlglot.parse(sql.strip(), read="postgres")
        if not stmts:
            return ["Empty parse result."]
        return []
    except Exception as exc:
        return [f"SQL syntax error: {exc}"]


def _cte_contract_block(depends_on: list[str], completed: dict) -> str:
    """
    Hard constraint block injected into HumanMessage.
    Lists the EXACT columns available from each dependency CTE.
    Uses actual parsed SELECT columns if available, else declared output_columns.
    """
    if not depends_on:
        return ""
    lines = [
        "HARD CONSTRAINT — CTE column contracts:",
        "You may ONLY reference columns listed here from each CTE.",
        "Any column not listed does not exist. Do not invent or guess column names.\n"
    ]
    for dep_id in depends_on:
        dep = completed.get(dep_id, {})

        # Prefer actual parsed columns — they are ground truth
        actual = _extract_select_columns(dep.get("sql", ""))
        declared = dep.get("output_columns") or []

        # Use actual if non-empty, else fall back to declared
        columns = actual if actual and "*" not in actual else declared

        if columns:
            lines.append(f"  CTE '{dep_id}' available columns: {', '.join(columns)}")
        else:
            lines.append(f"  CTE '{dep_id}': columns unknown — infer from intent: {dep.get('intent', '')}")

    return "\n".join(lines)


def _history_text(history: list[dict]) -> str:
    lines = []
    for turn in history[-3:]:
        if isinstance(turn, dict):
            lines.append(f"User: {turn.get('query', '')}\nSQL: {turn.get('sql', '')}")
    return "\n\n".join(lines) if lines else "None"


def _generate_node_sql(
    node: dict,
    completed: dict,
    state: AgentState,
    error_feedback: str = "",
    global_error: str = "",
) -> str:
    is_recursive = _is_recursive_node(node)
    schema_text = _schema_for_node(node.get("tables", []), state)
    semantic_facts = "\n".join(f"- {f}" for f in state.get("semantic_facts", [])) or "None"
    procedural_rules = "\n".join(f"- {r}" for r in state.get("procedural_rules", [])) or "None"
    history_text = _history_text(state.get("conversation_history", []))
    output_cols = node.get("output_columns") or []

    retry_block = ""
    if error_feedback:
        retry_block = f"\n\nFix this error from previous attempt:\n{error_feedback}"
    if global_error:
        retry_block += f"\n\nGlobal assembly error:\n{global_error}"

    # Hard constraint block — injected into HumanMessage, not system prompt
    cte_contract = _cte_contract_block(node.get("depends_on", []), completed)

    output_constraint = (
        f"HARD CONSTRAINT — your SELECT must output EXACTLY these column aliases: "
        f"{', '.join(output_cols)}. "
        f"No more, no less. Use these exact names."
        if output_cols else ""
    )

    if is_recursive:
        return_instruction = (
            "This node requires a recursive hierarchy traversal.\n"
            "Return a WITH RECURSIVE ... SELECT block and nothing else — "
            "no outer WITH, no semicolon.\n"
            "Structure:\n"
            "WITH RECURSIVE <name>(cols...) AS (\n"
            "  SELECT ... FROM <table> WHERE parent_col IS NULL  -- base case\n"
            "  UNION ALL\n"
            "  SELECT ... FROM <table> JOIN <name> ON ...        -- recursive step\n"
            ")\n"
            "SELECT ... FROM <name>"
        )
    else:
        return_instruction = (
            "Return ONLY a raw SELECT statement — "
            "no WITH clause, no markdown, no explanation."
        )

    messages = [
        SystemMessage(
            content=(
                f"{return_instruction}\n\n"
                "Rules:\n"
                "1. Use ONLY columns that exist in the schema or are listed in the CTE contracts below.\n"
                "2. Use LOWER() for case-insensitive string filters.\n"
                "3. Year filters: if 'data_year_range' CTE is available use "
                "EXTRACT(YEAR FROM <col>) = (SELECT data_year FROM data_year_range). "
                "Never use CURRENT_DATE when data_year_range is available.\n"
                "4. Lifetime spend = ALL orders, no year filter.\n"
                "5. GROUP BY required for any aggregation.\n"
                "6. Top-1-per-group: ROW_NUMBER() OVER (PARTITION BY <g> ORDER BY <m> DESC), "
                "then WHERE row_num = 1. Never LIMIT for per-group.\n"
                "7. Always qualify columns with table or CTE name.\n"
                "8. Never use ORDER BY inside a CTE body. ORDER BY is only valid in the final SELECT. If ranking is needed, use ROW_NUMBER() OVER (ORDER BY ...) instead.\n\n"
                "9. For category hierarchy spend rollup: "
                "join category_tree using ON p.category_id = ct.category_id (leaf match), "
                "then GROUP BY ct.ancestor_category_id (root rollup). "
                "NEVER write ON p.category_id = ct.ancestor_category_id — "
                "ancestor_category_id is the grouping target, not the join key.\n"
                f"Reusable schema facts:\n{semantic_facts}\n\n"
                f"Procedural rules:\n{procedural_rules}"
            )
        ),
        HumanMessage(
            content=(
                f"Conversation history:\n{history_text}\n\n"
                f"Intent: {node['intent']}\n\n"
                + (f"{output_constraint}\n\n" if output_constraint else "")
                + (f"{cte_contract}\n\n" if cte_contract else "")
                + f"Database schema:\n{schema_text}"
                + (f"\n\n{retry_block}" if retry_block else "")
            )
        ),
    ]

    response = llm.invoke(messages)
    return _strip_sql_fence(response.content)


# ------------------------------------------------------------------
# Main node
# ------------------------------------------------------------------

def sub_query_processor(state: AgentState) -> AgentState:
    sub_queries = list(state.get("sub_queries", []))
    if not sub_queries:
        return {
            **state,
            "is_valid": False,
            "plan_errors": ["Decomposition produced no sub-queries."],
        }

    global_error = ""
    if state.get("retry_count", 0) > 0 and state.get("validation_errors"):
        global_error = "; ".join(state["validation_errors"])

    completed: dict[str, dict] = {}

    for i, node in enumerate(sub_queries):
        error_feedback = ""
        sql = ""

        for attempt in range(MAX_NODE_RETRIES + 1):
            sql = _generate_node_sql(node, completed, state, error_feedback, global_error)

            # Layer 1: syntax check
            syntax_errors = _validate_syntax(sql)
            if syntax_errors:
                error_feedback = "; ".join(syntax_errors)
                logger.debug("node '%s' syntax error attempt %d: %s", node["id"], attempt + 1, error_feedback)
                continue

            # Layer 2: output column contract check
            declared = node.get("output_columns") or []
            col_errors = _validate_output_columns(sql, declared)
            if col_errors:
                error_feedback = "; ".join(col_errors)
                logger.debug("node '%s' column contract error attempt %d: %s", node["id"], attempt + 1, error_feedback)
                continue

            logger.debug("node '%s' passed all checks on attempt %d", node["id"], attempt + 1)
            break

        node_done = {**node, "sql": sql, "is_recursive": _is_recursive_node(node)}
        sub_queries[i] = node_done
        completed[node["id"]] = node_done

    return {
        **state,
        "sub_queries": sub_queries,
        "validation_errors": [],
        "plan_errors": [],
        "is_valid": True,
    }
