"""
Processes each sub-query node in the decomposition DAG.
For each node:
  - Nodes with depends_on=[] generate SQL against actual DB tables.
  - Nodes with depends_on=[...] generate SQL referencing named CTEs.
Uses llm (Groq 8B) — one call per node, nodes are simple by design.
Retries each node up to MAX_NODE_RETRIES times on syntax error.
"""
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage
from llm import llm
from state import AgentState

try:
    import sqlglot
except ImportError:
    sqlglot = None

logger = logging.getLogger(__name__)
MAX_NODE_RETRIES = 2


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _strip_sql_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:sql)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


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


def _cte_descriptions(depends_on: list[str], completed: dict) -> str:
    if not depends_on:
        return ""
    lines = ["Available CTEs (already computed, reference by name in FROM):"]
    for dep_id in depends_on:
        dep = completed.get(dep_id, {})
        lines.append(f"\n  CTE '{dep_id}':")
        lines.append(f"    Purpose: {dep.get('intent', 'unknown')}")
        sql = dep.get("sql", "")
        if sql:
            # Surface SELECT columns from the CTE for reference
            m = re.search(r"(?i)select\s+(.+?)\s+from", sql, re.DOTALL)
            if m:
                col_hint = re.sub(r"\s+", " ", m.group(1))[:300]
                lines.append(f"    Columns: {col_hint}")
    return "\n".join(lines)


def _validate_syntax(sql: str) -> list[str]:
    if sqlglot is None or not sql.strip():
        return []
    try:
        stmts = sqlglot.parse(sql, read="postgres")
        if len(stmts) != 1:
            return [f"Expected 1 statement, got {len(stmts)}."]
        return []
    except Exception as exc:
        return [f"SQL syntax error: {exc}"]


def _generate_node_sql(
    node: dict,
    completed: dict,
    state: AgentState,
    error_feedback: str = "",
    global_error: str = "",
) -> str:
    schema_text = _schema_for_node(node.get("tables", []), state)
    cte_text = _cte_descriptions(node.get("depends_on", []), completed)
    semantic_facts = "\n".join(f"- {f}" for f in state.get("semantic_facts", [])) or "None"

    retry_block = ""
    if error_feedback:
        retry_block = f"\n\nPrevious attempt error — fix this:\n{error_feedback}"
    if global_error:
        retry_block += f"\n\nGlobal assembly error from prior attempt:\n{global_error}"

    messages = [
        SystemMessage(
            content=(
                "You are a PostgreSQL SQL generator for one atomic sub-query.\n"
                "Write a single SELECT statement satisfying the given intent.\n\n"
                "Rules:\n"
                "1. Return ONLY raw SQL — no markdown, no explanation, no WITH clause.\n"
                "2. Use ONLY tables/columns listed in the schema. Never invent columns.\n"
                "3. If CTEs are listed, reference them by name in FROM — do not re-derive their data.\n"
                "4. Use LOWER() for case-insensitive string filters.\n"
                "5. Use EXTRACT(YEAR FROM <date_col>) = EXTRACT(YEAR FROM CURRENT_DATE) for current-year filters.\n"
                "6. Use GROUP BY for any aggregation.\n"
                "7. For top-1-per-group, use ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ... DESC) in a subquery.\n"
                "8. Always qualify column names with their table or CTE name.\n\n"
                f"Reusable database facts:\n{semantic_facts}"
            )
        ),
        HumanMessage(
            content=(
                f"Intent: {node['intent']}\n\n"
                f"Database schema:\n{schema_text}\n\n"
                f"{cte_text}"
                f"{retry_block}"
            )
        ),
    ]

    response = llm.invoke(messages)
    return _strip_sql_fence(response.content)


# ------------------------------------------------------------------
# Main node
# ------------------------------------------------------------------

def sub_query_processor(state: AgentState) -> AgentState:
    """
    Iterate through sub_queries DAG in order.
    Generate and syntax-validate SQL for each node.
    Store results back in sub_queries list.
    On retry (retry_count > 0), regenerate all nodes with error context.
    """
    sub_queries = list(state.get("sub_queries", []))
    if not sub_queries:
        return {
            **state,
            "is_valid": False,
            "plan_errors": ["Decomposition produced no sub-queries."],
        }

    # Pass global error context on retry
    global_error = ""
    if state.get("retry_count", 0) > 0 and state.get("validation_errors"):
        global_error = "; ".join(state["validation_errors"])

    completed: dict[str, dict] = {}

    for i, node in enumerate(sub_queries):
        error_feedback = ""
        sql = ""

        for attempt in range(MAX_NODE_RETRIES + 1):
            sql = _generate_node_sql(node, completed, state, error_feedback, global_error)
            syntax_errors = _validate_syntax(sql)
            if not syntax_errors:
                logger.debug("node '%s' SQL OK on attempt %d", node["id"], attempt + 1)
                break
            error_feedback = "; ".join(syntax_errors)
            logger.debug("node '%s' syntax error attempt %d: %s", node["id"], attempt + 1, error_feedback)

        node_done = {**node, "sql": sql}
        sub_queries[i] = node_done
        completed[node["id"]] = node_done

    return {
        **state,
        "sub_queries": sub_queries,
        "validation_errors": [],
        "plan_errors": [],
        "is_valid": True,
    }
