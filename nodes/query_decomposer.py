"""
Decomposes a tier 3/4 query into an ordered DAG of atomic sub-intents.
Uses llm_strong (Gemini) — one call per complex query.

Each node in the DAG is simple enough for the sub_query_processor to plan
and generate SQL independently. Nodes with depends_on=[] query actual DB
tables. Nodes with depends_on=[...] reference the named CTEs from prior nodes.
"""
import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage
from llm import llm_strong
from state import AgentState

logger = logging.getLogger(__name__)


def _strip_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _schema_summary(state: AgentState) -> str:
    lines = []
    for table, columns in state.get("schema", {}).items():
        col_names = ", ".join(c["name"] for c in columns)
        fks = [
            f"{c['name']} → {c['foreign_key']}"
            for c in columns if c.get("foreign_key")
        ]
        fk_str = f"  FK: {', '.join(fks)}" if fks else ""
        lines.append(f"  {table}({col_names}){fk_str}")
    return "\n".join(lines)


def query_decomposer(state: AgentState) -> AgentState:
    messages = [
        SystemMessage(
            content=(
                "You are a query decomposition expert for a SQL agent.\n"
                "Break a complex analytical query into an ordered list of simple, "
                "atomic sub-queries that together answer the original question.\n\n"
                "Rules:\n"
                "1. Each node must have a unique snake_case id (e.g. 'base_customers', 'spend_per_category').\n"
                "2. The LAST node id must always be 'final'.\n"
                "3. depends_on lists ids of nodes this node references as CTEs. "
                "Nodes with empty depends_on query actual database tables.\n"
                "4. tables lists ONLY actual database table names needed. "
                "If a node only references CTEs, tables must be [].\n"
                "5. Each node intent must be precise and self-contained — "
                "enough for a small LLM to write correct SQL for it alone.\n"
                "6. Keep it minimal: 3–5 nodes is typical. Never add a node unless necessary.\n"
                "7. Ordering: leaves (no dependencies) first, 'final' last.\n\n"
                "Return ONLY valid JSON — a list of node objects.\n"
                "Example shape:\n"
                "[\n"
                '  {"id": "eligible_customers", "intent": "customers from Pune with 3+ orders in current year — return customer_id, full name", "tables": ["customers", "orders"], "depends_on": []},\n'
                '  {"id": "category_spend", "intent": "total spend per customer per category this year — return customer_id, category_name, total_spend", "tables": ["orders", "order_items", "products", "categories"], "depends_on": []},\n'
                '  {"id": "top_category", "intent": "for each customer, the category with highest total_spend — return customer_id, category_name", "tables": [], "depends_on": ["category_spend"]},\n'
                '  {"id": "final", "intent": "join eligible_customers with top_category, filter where total_spend > avg(total_spend) across all eligible customers, sort by total_spend desc", "tables": [], "depends_on": ["eligible_customers", "category_spend", "top_category"]}\n'
                "]"
            )
        ),
        HumanMessage(
            content=(
                f"Database schema:\n{_schema_summary(state)}\n\n"
                f"User query: {state['user_query']}"
            )
        ),
    ]

    response = llm_strong.invoke(messages)
    logger.debug("decomposer response: %s", response.content)

    try:
        nodes = json.loads(_strip_fence(response.content))
        if not isinstance(nodes, list):
            nodes = []
    except (json.JSONDecodeError, Exception):
        nodes = []

    validated = []
    for node in nodes:
        if isinstance(node, dict) and node.get("id") and node.get("intent"):
            validated.append({
                "id": str(node["id"]),
                "intent": str(node["intent"]),
                "tables": node.get("tables") or [],
                "depends_on": node.get("depends_on") or [],
                "sql": None,
            })

    logger.debug("decomposed into %d nodes: %s", len(validated), [n["id"] for n in validated])

    return {
        **state,
        "sub_queries": validated,
        # Reset retry state for the complex path
        "retry_count": 0,
        "validation_errors": [],
        "generated_sql": "",
    }
