"""
Query planner for tier 1 and 2 queries only.
Tier 3/4 queries never reach this node — they go through query_decomposer.
Produces a structured JSON plan for simple single-table or basic join+aggregation queries.
"""
import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage
from llm import llm
from state import AgentState

logger = logging.getLogger(__name__)


def _parse_json(text: str) -> dict:
    text = text.strip()

    # Strip a code fence if the response is purely fenced JSON
    stripped = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped).strip()
    for candidate in (stripped, text):
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    # LLM added preamble/postamble — extract the first {...} block
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    return {}


def _schema_text(state: AgentState) -> str:
    lines = []
    for table, columns in state.get("schema", {}).items():
        lines.append(f"Table: {table}")
        for col in columns:
            annot = []
            if col.get("primary_key"):
                annot.append("PK")
            if col.get("foreign_key"):
                annot.append(f"FK → {col['foreign_key']}")
            if col.get("sample_values"):
                samples = ", ".join(str(v) for v in col["sample_values"][:8])
                annot.append(f"samples: {samples}")
            suffix = f" ({'; '.join(annot)})" if annot else ""
            lines.append(f"  {col['name']} {col['type']}{suffix}")
    return "\n".join(lines)


def _history_text(history: list[dict]) -> str:
    lines = []
    for turn in history[-2:]:
        if isinstance(turn, dict):
            lines.append(f"User: {turn.get('query', '')}\nSQL: {turn.get('sql', '')}")
    return "\n\n".join(lines) if lines else "None"


def _retry_feedback(state: AgentState) -> str:
    errors = (
        state.get("plan_errors", [])
        + state.get("validation_errors", [])
        + state.get("semantic_validation_errors", [])
    )
    if not errors and not state.get("execution_error"):
        return ""
    parts = []
    if errors:
        parts.append(f"Errors from previous attempt: {errors}")
    if state.get("execution_error"):
        parts.append(f"Execution error: {state['execution_error']}")
    parts.append("Revise the plan to fix these.")
    return "\n\n" + "\n".join(parts)


def _normalise_plan(raw: dict, state: AgentState) -> dict:
    valid_map = {t.lower(): t for t in state.get("table_names", [])}
    raw_tables = raw.get("tables") or []
    if isinstance(raw_tables, str):
        raw_tables = [raw_tables]
    tables = [
        valid_map[t.strip().lower()]
        for t in raw_tables
        if isinstance(t, str) and t.strip().lower() in valid_map
    ]
    if not tables:
        tables = state.get("relevant_tables", [])

    return {
        "intent": raw.get("intent") or state["user_query"],
        "tables": tables,
        "metrics": raw.get("metrics") or [],
        "dimensions": raw.get("dimensions") or [],
        "filters": raw.get("filters") or [],
        "joins": raw.get("joins") or [],
        "aggregation_scope": raw.get("aggregation_scope") or "unspecified",
        "ranking": raw.get("ranking"),
        "time_scope": raw.get("time_scope"),
        "hierarchy": raw.get("hierarchy"),
        "output": raw.get("output") or [],
        "assumptions": raw.get("assumptions") or [],
    }


def query_planner(state: AgentState) -> AgentState:
    messages = [
        SystemMessage(
            content=(
                "You plan simple SQL queries (single-table lookups or basic multi-table aggregations).\n"
                "You will NEVER receive queries needing HAVING on aggregations, "
                "window functions, or multi-level subqueries — those are handled separately.\n\n"
                "STRICT RULES — violating any of these is a failure:\n"
                "1. metrics MUST be a non-empty list for any aggregation query (total, count, sum, average).\n"
                "2. dimensions MUST be a non-empty list whenever the query groups results (per city, per category, each X).\n"
                "3. joins MUST have one entry per additional table beyond the first. Never leave joins empty if tables has 2+ entries.\n"
                "4. Every field MUST be table-qualified: orders.customer_id not customer_id.\n"
                "5. Revenue or price always requires joining through order_items to products — never assume price lives on orders.\n\n"
                "WORKED EXAMPLE — query: 'total revenue by city ranked highest to lowest'\n"
                "WARNING: the field names below (e.g. customers.id) are illustrative placeholders only.\n"
                "For the actual query you MUST replace every field name with the exact column name "
                "from the schema provided. Read the schema FK annotations to determine join columns.\n"
                "{\n"
                '  "intent": "Total revenue per city, ranked highest to lowest",\n'
                '  "tables": ["customers", "orders", "order_items", "products"],\n'
                '  "metrics": [{"name": "total_revenue", "type": "sum", "field": "<quantity_col> * <price_col> — use exact column names from schema"}],\n'
                '  "dimensions": [{"field": "customers.<city_col>", "purpose": "group_by"}],\n'
                '  "filters": [],\n'
                '  "joins": [\n'
                '    {"left": "customers.<pk_col>", "right": "orders.<fk_to_customers>", "type": "inner"},\n'
                '    {"left": "orders.<pk_col>", "right": "order_items.<fk_to_orders>", "type": "inner"},\n'
                '    {"left": "order_items.<fk_to_products>", "right": "products.<pk_col>", "type": "inner"}\n'
                '  ],\n'
                '  "aggregation_scope": "per_group",\n'
                '  "ranking": {"order_by": "total_revenue", "direction": "desc", "limit": null},\n'
                '  "time_scope": null,\n'
                '  "output": ["customers.<city_col>", "total_revenue"],\n'
                '  "assumptions": []\n'
                "}\n\n"
                "CRITICAL: resolve every <placeholder> using the schema below before returning JSON. "
                "Join fields must exactly match the FK annotations shown in the schema. "
                "Set time_scope to null if no date range is needed. "
                "Return ONLY valid JSON."
            )
        ),
        HumanMessage(
            content=(
                f"Schema:\n{_schema_text(state)}\n\n"
                f"Memory facts:\n{state.get('semantic_facts', [])}\n\n"
                f"Recent history:\n{_history_text(state.get('conversation_history', []))}\n\n"
                f"Query: {state['user_query']}"
                f"{_retry_feedback(state)}"
            )
        ),
    ]

    response = llm.invoke(messages)
    logger.debug("planner response: %s", response.content)

    raw = _parse_json(response.content)
    plan = _normalise_plan(raw, state)

    return {
        **state,
        "query_plan": plan,
        "query_intent": plan["intent"],
        "relevant_tables": plan["tables"],
        "plan_errors": [],
        "validation_errors": [],
        "semantic_validation_errors": [],
        "generated_sql": "",
        "execution_error": None,
    }
