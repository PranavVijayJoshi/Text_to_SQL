# nodes/query_cache.py
import hashlib
import json
import time
from state import AgentState


# In-memory cache for this session.
# In production, swap this to Redis or another shared cache if you run multiple workers.
_cache: dict[str, dict] = {}
CACHE_TTL_SECONDS = 3600


def _query_hash(state: AgentState) -> str:
    history = state.get("conversation_history", [])
    recent_history = history[-3:] if isinstance(history, list) else []

    payload = {
        "user_id": state.get("user_id"),
        "query": " ".join(state.get("user_query", "").lower().strip().split()),
        "query_plan": state.get("query_plan", {}),
        "db": state.get("db_connection_string"),
        "history": recent_history,
        "tables": sorted(state.get("relevant_tables", [])),
    }

    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def query_cache(state: AgentState) -> AgentState:
    key = _query_hash(state)
    now = time.time()
    entry = _cache.get(key)

    if entry and (now - entry["timestamp"]) < CACHE_TTL_SECONDS:
        print(f"[cache hit] returning cached result for: {state['user_query']}")
        history = list(state.get("conversation_history", []))
        history.append(
            {
                "query": state["user_query"],
                "summary": entry["final_answer"],
                "sql": entry["sql"],
            }
        )
        return {
            **state,
            "generated_sql": entry["sql"],
            "relevant_tables": entry["relevant_tables"],
            "query_intent": entry["intent"],
            "query_plan": entry.get("query_plan", {}),
            "results": entry["results"],
            "final_answer": entry["final_answer"],
            "status": "success",
            "error_message": None,
            "conversation_history": history,
            "cache_hit": True,
        }

    return {**state, "cache_hit": False}


def update_cache(state: AgentState) -> AgentState:
    """Store only successful query executions."""
    if state.get("execution_error") is not None:
        return state
    if not state.get("final_answer"):
        return state

    key = _query_hash(state)
    _cache[key] = {
        "timestamp": time.time(),
        "sql": state["generated_sql"],
        "relevant_tables": state["relevant_tables"],
        "intent": state["query_intent"],
        "query_plan": state.get("query_plan", {}),
        "results": state["results"],
        "final_answer": state["final_answer"],
    }
    return state
