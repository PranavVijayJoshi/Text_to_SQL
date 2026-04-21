"""
Decomposes a tier 3/4 query into an ordered DAG of atomic sub-intents.
Uses llm_strong — one call per complex query.

Receives full memory context (semantic facts, past episodes, procedural rules)
and conversation history so pronoun resolution and learned schema facts
apply to complex queries just as they do on the fast path.
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


def _history_text(history: list[dict]) -> str:
    lines = []
    for turn in history[-3:]:
        if isinstance(turn, dict):
            lines.append(f"User: {turn.get('query', '')}\nSQL: {turn.get('sql', '')}")
    return "\n\n".join(lines) if lines else "None"


def _memory_block(state: AgentState) -> str:
    facts = "\n".join(f"- {f}" for f in state.get("semantic_facts", [])) or "None"
    rules = "\n".join(f"- {r}" for r in state.get("procedural_rules", [])) or "None"
    episodes = "\n".join(f"- {e}" for e in state.get("past_episodes", [])) or "None"
    return (
        f"Reusable schema facts:\n{facts}\n\n"
        f"Procedural rules from past failures:\n{rules}\n\n"
        f"Similar past queries:\n{episodes}"
    )


def query_decomposer(state: AgentState) -> AgentState:
    messages = [
        SystemMessage(
            content=(
                "You are a query decomposition expert for a SQL agent.\n"
                "Break a complex analytical query into an ordered list of simple, "
                "atomic sub-queries that together answer the original question.\n\n"
                "Use the conversation history to resolve any pronouns or references "
                "from previous turns (e.g. 'them', 'those customers', 'same period').\n"
                "Use the schema facts and procedural rules to avoid known mistakes.\n\n"
                "Rules:\n"
                "1. Each node must have a unique snake_case id.\n"
                "2. The LAST node id must always be 'final'.\n"
                "3. depends_on lists ids of nodes this node references as CTEs. "
                "Nodes with empty depends_on query actual database tables.\n"
                "4. tables lists ONLY actual database table names. "
                "If a node only references CTEs, tables must be [].\n"
                "5. Each node intent must be precise and self-contained.\n"
                "6. Keep it minimal: 3–5 nodes is typical.\n"
                "7. Ordering: leaves (no dependencies) first, 'final' last.\n\n"
                "Return ONLY valid JSON — a list of node objects.\n"
                "Example shape:\n"
                "[\n"
                '  {"id": "eligible_customers", "intent": "customers from Pune with 3+ orders '
                'in current year — return customer_id, full name", "tables": ["customers", "orders"], "depends_on": []},\n'
                '  {"id": "final", "intent": "return eligible_customers sorted by name", '
                '"tables": [], "depends_on": ["eligible_customers"]}\n'
                "]"
            )
        ),
        HumanMessage(
            content=(
                f"Database schema:\n{_schema_summary(state)}\n\n"
                f"{_memory_block(state)}\n\n"
                f"Conversation history:\n{_history_text(state.get('conversation_history', []))}\n\n"
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
        "retry_count": 0,
        "validation_errors": [],
        "generated_sql": "",
    }
