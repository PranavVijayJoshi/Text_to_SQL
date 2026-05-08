"""
Decomposes a tier 3/4 query into an ordered DAG of atomic sub-intents.
Uses llm_strong — one call per complex query.

Each node declares output_columns — an explicit contract of what columns
its SELECT produces. Dependent nodes use these declared columns, not regex
guesses, eliminating cross-node column name mismatch bugs.
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
                "5. Each node intent must be precise and self-contained. If a dependency CTE has already applied a filter (city, year, month), the dependent node must NOT re-apply the same filter. State in the node intent exactly which filters are already enforced by dependencies.\n"
                "6. Keep it minimal: 3–6 nodes is typical.\n"
                "7. Ordering: leaves (no dependencies) first, 'final' last.\n"
                "8. output_columns is REQUIRED for every node — list the exact column aliases "
                "the node's SELECT will produce (e.g. ['customer_id', 'total_spend']). "
                "Dependent nodes MUST only reference column names declared in their dependency's output_columns. "
                "This is a hard contract — mismatched column names cause SQL errors.\n"
                "9. Any node that finds the top/maximum value per group (e.g. top category per customer, "
                "highest spend per city) MUST state in its intent: 'use ROW_NUMBER() OVER "
                "(PARTITION BY <group_col> ORDER BY <metric> DESC), return only row_num = 1'. "
                "Never use LIMIT for per-group selection."
                "For top-1-per-group selection, the node intent must explicitly state: use ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY category_spend DESC), filter WHERE row_num = 1. Never use MAX() with IN() — it does not handle ties and can return multiple rows per group.\n"
                "10. If the query involves sub-categories, category hierarchy, or ancestor/descendant "
                "relationships, you MUST include a dedicated node with id 'category_tree' that uses "
                "WITH RECURSIVE on the self-referencing parent column to flatten the full hierarchy "
                "(each row = a leaf category_id paired with every ancestor category_id and name). "
                "This node must appear before any category-level spend aggregation node and must be "
                "listed in that node's depends_on. Set tables to the categories table."
                "When a category_tree node is used for spend aggregation, the grouping dimension must be ancestor_category_id, not category_id, so spend rolls up to root level."
                "When joining category_tree to products, always join on p.category_id = ct.category_id (leaf match), then GROUP BY ct.ancestor_category_id to roll spend up to root. Never join on ct.ancestor_category_id — that column is the grouping target, not the join key.\n"
                "CORRECT example for category spend rollup node intent:\n"
                "'For each customer, compute total spend per ROOT category by joining orders → order_items → products → category_tree ON p.category_id = ct.category_id (leaf match), then GROUP BY customer_id, ct.ancestor_category_id (root rollup). Return customer_id, ancestor_category_id, SUM(quantity * unit_price) AS category_spend.'\n"
                "WRONG — never write: JOIN category_tree ON p.category_id = ct.ancestor_category_id\n"
                "The ancestor_category_id column is the GROUP BY target, never the join key.\n"
                "11. Whenever the query involves a time filter (this year, current year, recent orders, "
                "different months this year), add a node with id 'data_year_range' as the very first node. "
                "Its intent must be: 'find the most recent year present in the orders table — "
                "return a single column named data_year'. "
                "Every subsequent node that filters by year MUST list 'data_year_range' in depends_on "
                "and use data_year_range.data_year for the year filter.\n"
                "12. Lifetime spend means ALL orders regardless of year — never apply a year filter "
                "to a lifetime spend calculation.\n\n"
                "Return ONLY valid JSON — a list of node objects.\n"
                "Example shape:\n"
                "[\n"
                '  {"id": "data_year_range", "intent": "find the most recent year in orders — '
                'return single column named data_year", "tables": ["orders"], "depends_on": [], '
                '"output_columns": ["data_year"]},\n'
                '  {"id": "eligible_customers", "intent": "customers from Pune with orders in 2+ '
                'distinct months — use data_year_range.data_year for year filter — '
                'return customer_id, first_name, last_name", '
                '"tables": ["customers", "orders"], "depends_on": ["data_year_range"], '
                '"output_columns": ["customer_id", "first_name", "last_name"]},\n'
                '  {"id": "final", "intent": "return eligible_customers sorted by name", '
                '"tables": [], "depends_on": ["eligible_customers"], '
                '"output_columns": ["first_name", "last_name"]}\n'
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
                "output_columns": node.get("output_columns") or [],
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
