import uuid
from langmem import create_memory_store_manager
from langchain_core.messages import HumanMessage, SystemMessage
from memory_store import store
from state import AgentState
from llm import llm

def memory_update(state:AgentState)-> AgentState:
    user_id = state.get("user_id", "default_user")
    #semantic_manager = create_memory_manager(store=store, namespace = "semantic")
    semantic_response = llm.invoke([
        SystemMessage(content=(
            "You are a database knowledge extractor. "
            "Given a SQL query and its outcome, identify reusable facts "
            "about the database schema that would help future queries.\n\n"
            "Good facts to extract:\n"
            "- Join paths: which columns connect tables (e.g. 'orders joins customers on customer_id')\n"
            "- Revenue/amount location: which column and table holds monetary values\n"
            "- Date column names and their formats\n"
            "- Primary key column names per table\n"
            "- Any non-obvious column names (e.g. 'product price is in products.price, not order_items')\n\n"
            "Do NOT extract facts that are obvious from the schema alone.\n"
            "If there are no new facts worth saving, reply with: NONE\n"
            "Otherwise list each fact on a new line, no bullet points."
        )),
        HumanMessage(content=(
            f"User query: {state['user_query']}\n"
            f"Tables used: {state['relevant_tables']}\n"
            f"Final SQL: {state['generated_sql']}\n"
            f"Execution error: {state.get('execution_error', 'none')}\n"
            f"Retries needed: {state['retry_count']}"
        )),
    ])
    if semantic_response.content.strip().upper() != "NONE":
        for fact in semantic_response.content.strip().split('\n'):
            fact=fact.strip("- ").strip()
            if fact:
                store.put(
                    ("semantic",),
                    str(uuid.uuid4()),
                    {"content":fact}
                )

    episode = (
        f"Query: {state['user_query']}\n"
        f"Intent: {state['query_intent']}\n"
        f"Tables: {', '.join(state['relevant_tables'])}\n"
        f"SQL: {state['generated_sql']}\n"
        f"Success: {state.get('execution_error') is None}\n"
        f"Retries: {state['retry_count']}"
    )
    store.put(
        ("episodes", user_id),
        str(uuid.uuid4()),
        {"content": episode}
    )
    
    if state["retry_count"] > 0 or state.get("execution_error"):
        procedural_response = llm.invoke([
            SystemMessage(content=(
                "You are a behavioural rule extractor for a SQL agent. "
                "Given a failed SQL run, write one reusable rule to prevent this.\n\n"
                "Examples:\n"
                "- 'Always add LIMIT 100 when user does not specify row count'\n"
                "- 'Use CTE instead of nested subqueries'\n\n"
                "If no rule can be learned, reply: NONE\n"
                "Otherwise state the rule in one sentence."
            )),
            HumanMessage(content=(
                f"User query: {state['user_query']}\n"
                f"Validation errors: {state['validation_errors']}\n"
                f"Execution error: {state.get('execution_error', 'none')}\n"
                f"Final SQL: {state['generated_sql']}"
            )),
        ])

        rule = procedural_response.content.strip()
        if rule.upper() != "NONE" and rule:
            store.put(
                ("procedural", user_id),
                str(uuid.uuid4()),
                {"content": rule}
            )

    return state

