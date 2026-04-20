import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from llm import llm
from state import AgentState


TEXT_TYPE_MARKERS = ("char", "text", "string", "enum", "citext", "user-defined")


def _string_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _text_columns(schema: dict[str, list[dict]]) -> set[str]:
    columns = set()
    for table, table_columns in schema.items():
        for column in table_columns:
            column_type = str(column.get("type", "")).lower()
            if any(marker in column_type for marker in TEXT_TYPE_MARKERS):
                column_name = column.get("name", "")
                if column_name:
                    columns.add(column_name.lower())
                    columns.add(f"{table.lower()}.{column_name.lower()}")
    return columns


def _make_text_filters_case_insensitive(sql: str, state: AgentState) -> str:
    text_columns = _text_columns(state.get("schema", {}))
    if not text_columns:
        return sql

    comparison_pattern = re.compile(
        r"(?<![\w.])(?P<lhs>(?:(?P<table>[A-Za-z_][A-Za-z0-9_]*)\.)?"
        r"(?P<column>[A-Za-z_][A-Za-z0-9_]*))\s*=\s*'(?P<value>(?:''|[^'])*)'",
        re.IGNORECASE,
    )

    def replace(match: re.Match) -> str:
        lhs = match.group("lhs")
        table = match.group("table")
        column = match.group("column")
        value = match.group("value").replace("''", "'")
        lookup_keys = {column.lower()}
        if table:
            lookup_keys.add(f"{table.lower()}.{column.lower()}")

        if not lookup_keys.intersection(text_columns):
            return match.group(0)

        return f"LOWER(CAST({lhs} AS TEXT)) = LOWER({_string_literal(value)})"

    return comparison_pattern.sub(replace, sql)


def _schema_text(state: AgentState) -> str:
    lines = []
    for table, columns in state["schema"].items():
        lines.append(f"Table: {table}")
        for column in columns:
            annotations = []
            if column.get("primary_key"):
                annotations.append("primary key")
            if column.get("foreign_key"):
                annotations.append(f"foreign key -> {column['foreign_key']}")
            if column.get("sample_values"):
                samples = ", ".join(str(value) for value in column["sample_values"][:10])
                annotations.append(f"sample values: {samples}")
            suffix = f" ({'; '.join(annotations)})" if annotations else ""
            lines.append(f"- {column['name']} {column['type']}{suffix}")
    return "\n".join(lines)


def _retry_context(state: AgentState) -> str:
    retry_parts = []
    if state.get("retry_count", 0) > 0:
        if state.get("generated_sql"):
            retry_parts.append(f"Previous SQL attempt:\n{state['generated_sql']}")
        if state.get("plan_errors"):
            retry_parts.append(f"Plan errors:\n{state['plan_errors']}")
        if state.get("validation_errors"):
            retry_parts.append(f"SQL validation errors:\n{state['validation_errors']}")
        if state.get("semantic_validation_errors"):
            retry_parts.append(f"Semantic SQL errors:\n{state['semantic_validation_errors']}")
        if state.get("execution_error"):
            retry_parts.append(f"Database execution error:\n{state['execution_error']}")

    if not retry_parts:
        return ""
    return "\n\n".join(retry_parts) + "\n\nFix the SQL while preserving the verified plan."


def sql_generator(state: AgentState) -> AgentState:
    """
    Compile the verified structured query plan into PostgreSQL.
    """
    plan_json = json.dumps(state.get("query_plan", {}), indent=2, default=str)
    semantic_block = (
        "\n".join(f"- {fact}" for fact in state.get("semantic_facts", []))
        or "None available"
    )
    procedural_block = (
        "\n".join(f"- {rule}" for rule in state.get("procedural_rules", []))
        or "None available"
    )
    episodic_block = (
        "\n".join(f"- {episode}" for episode in state.get("past_episodes", []))
        or "None available"
    )

    messages = [
        SystemMessage(
            content=(
                "You are a PostgreSQL compiler for a natural-language-to-SQL agent.\n"
                "Your job is to compile the structured query plan into SQL. "
                "Do not reinterpret the original user question when it conflicts with the plan.\n\n"
                "Rules:\n"
                "1. Return only one read-only SELECT query.\n"
                "2. Use only tables and columns in the schema context.\n"
                "3. Never reference a column on a table unless that exact column is listed under that table.\n"
                "4. Use foreign keys in the schema context to reach attributes on dimension tables. "
                "For example, if category_id is on products, join order_items to products before using category_id.\n"
                "5. Use the plan's metrics, filters, dimensions, joins, ranking, time scope, and hierarchy.\n"
                "6. For text/category filters, use case-insensitive comparison.\n"
                "7. Use GROUP BY or window functions when aggregation_scope is per-group/per-user/per-customer/per-time-period.\n"
                "8. Use WITH RECURSIVE only when the plan's hierarchy.required is true.\n"
                "9. Do not reference SELECT aliases in the same SELECT or HAVING scope; use another CTE/subquery.\n"
                "10. Do not reference an outer query alias from a subquery in FROM unless using LATERAL; prefer independent CTEs.\n"
                "11. For rolling month analysis, build non-overlapping month buckets with date_trunc('month', order_date).\n"
                "12. To prove a customer did not buy a category in a period, use NOT EXISTS rather than expecting a zero row.\n"
                "13. Return ONLY raw SQL. No markdown, no explanation.\n\n"
                "Reusable database facts:\n"
                f"{semantic_block}\n\n"
                "Procedural rules:\n"
                f"{procedural_block}\n\n"
                "Similar past queries:\n"
                f"{episodic_block}"
            )
        ),
        HumanMessage(
            content=(
                f"Schema context:\n{_schema_text(state)}\n\n"
                f"Structured query plan:\n{plan_json}\n\n"
                f"Original user query:\n{state['user_query']}\n\n"
                f"{_retry_context(state)}"
            )
        ),
    ]

    response = llm.invoke(messages)
    sql = response.content.strip()

    sql = re.sub(r"^```(?:sql)?\s*", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\s*```$", "", sql)
    sql = _make_text_filters_case_insensitive(sql.strip(), state)

    return {
        **state,
        "generated_sql": sql,
        "execution_error": None,
    }
