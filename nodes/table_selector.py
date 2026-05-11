import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage
from llm import llm
from state import AgentState


logger = logging.getLogger(__name__)


def _parse_json(text: str) -> dict:
    text = text.strip()

    stripped = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped).strip()
    for candidate in (stripped, text):
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    return {}


def _metadata_text(state: AgentState) -> str:
    return "\n".join(
        f"- {table}" + (f": {hint}" if hint else "")
        for table, hint in state.get("table_metadata", {}).items()
    )


def _normalise_tables(tables, state: AgentState) -> list[str]:
    valid_table_map = {table.lower(): table for table in state.get("table_names", [])}
    if isinstance(tables, str):
        tables = [tables]

    selected = [
        valid_table_map[table.strip().lower()]
        for table in tables or []
        if isinstance(table, str) and table.strip().lower() in valid_table_map
    ]

    query_lower = state["user_query"].lower()
    bridge_hints = {
        "category": ["products", "categories"],
        "product": ["products"],
        "purchase": ["orders", "order_items"],
        "spend": ["orders", "order_items"],
        "order": ["orders", "order_items"],
        "customer": ["customers"],
    }
    for keyword, required_tables in bridge_hints.items():
        if keyword in query_lower:
            for table in required_tables:
                if table in valid_table_map and valid_table_map[table] not in selected:
                    selected.append(valid_table_map[table])

    return selected or state.get("table_names", [])[:5]


def table_selector(state: AgentState) -> AgentState:
    retry_feedback = ""
    errors = (
        state.get("plan_errors", [])
        + state.get("validation_errors", [])
        + state.get("semantic_validation_errors", [])
    )
    if errors or state.get("execution_error"):
        retry_feedback = (
            "\n\nPrevious failure feedback:\n"
            f"Errors: {errors}\n"
            f"Execution error: {state.get('execution_error') or ''}\n"
            "If a field is reached through a bridge/dimension table, include that table."
        )

    messages = [
        SystemMessage(
            content=(
                "You select the database tables needed before detailed schema is loaded.\n"
                "Do not write SQL and do not create a full query plan.\n"
                "Include bridge and dimension tables needed for joins. For example, if the user asks "
                "about product categories and purchases, include orders, order_items, products, and categories.\n\n"
                "Return only valid JSON in this shape:\n"
                '{"intent": "self-contained intent", "tables": ["customers", "orders"]}'
            )
        ),
        HumanMessage(
            content=(
                f"Available tables:\n{_metadata_text(state)}\n\n"
                f"User query: {state['user_query']}"
                f"{retry_feedback}"
            )
        ),
    ]

    response = llm.invoke(messages)
    parsed = _parse_json(response.content)
    logger.debug("table selector response: %s", response.content)

    relevant_tables = _normalise_tables(parsed.get("tables", []), state)
    return {
        **state,
        "query_intent": parsed.get("intent") or state["user_query"],
        "relevant_tables": relevant_tables,
    }
