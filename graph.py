"""
Graph definition.

Fast path  (tier 1/2): planner → validator → SQL generator → validator → executor
Complex path (tier 3/4): decomposer → sub-query processor → assembler → validator → executor

Both paths share: intake, memory, cache, executor, formatter, memory-update.
"""
from langgraph.graph import END, START, StateGraph

from config import MAX_RETRIES
from nodes.answer_formatter import answer_formatter
from nodes.complexity_classifier import complexity_classifier
from nodes.guardrail_analyzer import guardrail_analyzer
from nodes.memory_retrieval import memory_retrieval
from nodes.memory_update import memory_update
from nodes.metadata_loader import metadata_loader
from nodes.plan_validator import plan_validator
from nodes.query_cache import query_cache, update_cache
from nodes.query_decomposer import query_decomposer
from nodes.query_executor import query_executor
from nodes.query_planner import query_planner
from nodes.schema_loader import schema_loader
from nodes.semantic_sql_validator import semantic_sql_validator
from nodes.sql_assembler import sql_assembler
from nodes.sql_generator import sql_generator
from nodes.sql_validator import sql_validator
from nodes.sub_query_processor import sub_query_processor
from nodes.table_selector import table_selector
from state import AgentState


# ------------------------------------------------------------------
# Routing helpers
# ------------------------------------------------------------------

def route_complexity(state: AgentState) -> str:
    """Route tier 1/2 to fast path, tier 3/4 to complex path."""
    return "complex" if state.get("complexity_tier", 1) >= 3 else "fast"


def check_cache(state: AgentState) -> str:
    return "end" if state.get("cache_hit") else "continue"


def check_relevance(state: AgentState) -> str:
    if not state.get("is_relevant", True):
        return "end"
    return "continue"


def _all_errors(state: AgentState) -> list[str]:
    seen, out = set(), []
    for e in (
        state.get("plan_errors", [])
        + state.get("validation_errors", [])
        + state.get("semantic_validation_errors", [])
    ):
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out


def route_plan_validation(state: AgentState) -> str:
    if state.get("is_valid"):
        return "continue"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return "failed"
    return "retry"


def route_sql_validation(state: AgentState) -> str:
    if state.get("is_valid"):
        return "continue"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return "failed"
    errors_text = " ".join(_all_errors(state)).lower()
    if any(k in errors_text for k in ("does not exist", "not found in schema", "unknown field", "missing from-clause")):
        return "replan"
    return "retry"


def route_complex_validation(state: AgentState) -> str:
    if state.get("is_valid"):
        return "continue"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return "failed"
    return "retry"


def after_fast_execution(state: AgentState) -> str:
    if state.get("execution_error") is None:
        return "success"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return "failed"
    return "retry"


def after_complex_execution(state: AgentState) -> str:
    if state.get("execution_error") is None:
        return "success"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return "failed"
    return "retry"


# ------------------------------------------------------------------
# Terminal response builders
# ------------------------------------------------------------------

def _irrelevance_response(state: AgentState) -> AgentState:
    return {
        **state,
        "final_answer": (
            f"Sorry, I can only help with database queries.\n"
            f"Reason: {state.get('rejection_reason', '')}"
        ),
        "status": "failed",
        "error_message": state.get("rejection_reason", ""),
    }


def validation_failure_response(state: AgentState) -> AgentState:
    errors = _all_errors(state)
    return {
        **state,
        "final_answer": (
            "I could not produce a valid SQL query after multiple attempts.\n\n"
            f"Errors:\n" + "\n".join(f"  • {e}" for e in errors)
        ),
        "status": "failed",
        "error_message": "; ".join(errors),
    }


def execution_failure_response(state: AgentState) -> AgentState:
    return {
        **state,
        "final_answer": (
            "I generated SQL but it failed during execution after multiple attempts.\n\n"
            f"Last SQL:\n{state.get('generated_sql', '')}\n\n"
            f"Error:\n{state.get('execution_error', '')}"
        ),
        "status": "failed",
        "error_message": state.get("execution_error", ""),
    }


def safe_memory_update(state: AgentState) -> AgentState:
    try:
        return memory_update(state)
    except Exception as exc:
        print(f"[memory_update error]: {exc}")
        return state


# ------------------------------------------------------------------
# Graph builder
# ------------------------------------------------------------------

def build_graph():
    g = StateGraph(AgentState)

    # ---------- Shared intake nodes ----------
    g.add_node("metadata_loader", metadata_loader)
    g.add_node("guardrail_analyzer", guardrail_analyzer)
    g.add_node("irrelevance_response", _irrelevance_response)
    g.add_node("memory_retrieval", memory_retrieval)
    g.add_node("table_selector", table_selector)
    g.add_node("schema_loader", schema_loader)
    g.add_node("complexity_classifier", complexity_classifier)

    # ---------- Fast path (tier 1/2) ----------
    g.add_node("query_cache", query_cache)
    g.add_node("query_planner", query_planner)
    g.add_node("plan_validator", plan_validator)
    g.add_node("sql_generator", sql_generator)
    g.add_node("sql_validator", sql_validator)
    g.add_node("semantic_sql_validator", semantic_sql_validator)
    g.add_node("fast_validation_failure", validation_failure_response)
    g.add_node("fast_executor", query_executor)
    g.add_node("fast_execution_failure", execution_failure_response)

    # ---------- Complex path (tier 3/4) ----------
    g.add_node("query_decomposer", query_decomposer)
    g.add_node("sub_query_processor", sub_query_processor)
    g.add_node("sql_assembler", sql_assembler)
    g.add_node("complex_sql_validator", sql_validator)           # same fn, different node
    g.add_node("complex_validation_failure", validation_failure_response)
    g.add_node("complex_executor", query_executor)
    g.add_node("complex_execution_failure", execution_failure_response)

    # ---------- Shared finish nodes ----------
    g.add_node("answer_formatter", answer_formatter)
    g.add_node("memory_update", safe_memory_update)
    g.add_node("update_cache", update_cache)

    # ================================================================
    # Edges — Intake
    # ================================================================
    g.add_edge(START, "metadata_loader")
    g.add_edge("metadata_loader", "guardrail_analyzer")
    g.add_conditional_edges(
        "guardrail_analyzer",
        check_relevance,
        {"continue": "memory_retrieval", "end": "irrelevance_response"},
    )
    g.add_edge("irrelevance_response", END)
    g.add_edge("memory_retrieval", "table_selector")
    g.add_edge("table_selector", "schema_loader")
    g.add_edge("schema_loader", "complexity_classifier")

    # ================================================================
    # Complexity router
    # ================================================================
    g.add_conditional_edges(
        "complexity_classifier",
        route_complexity,
        {"fast": "query_cache", "complex": "query_decomposer"},
    )

    # ================================================================
    # Fast path
    # ================================================================
    g.add_conditional_edges(
        "query_cache",
        check_cache,
        {"end": END, "continue": "query_planner"},
    )
    g.add_edge("query_planner", "plan_validator")
    g.add_conditional_edges(
        "plan_validator",
        route_plan_validation,
        {"continue": "sql_generator", "retry": "query_planner", "failed": "fast_validation_failure"},
    )
    g.add_edge("sql_generator", "sql_validator")
    g.add_conditional_edges(
        "sql_validator",
        route_sql_validation,
        {
            "continue": "semantic_sql_validator",
            "retry": "sql_generator",
            "replan": "table_selector",
            "failed": "fast_validation_failure",
        },
    )
    g.add_conditional_edges(
        "semantic_sql_validator",
        route_plan_validation,
        {"continue": "fast_executor", "retry": "sql_generator", "failed": "fast_validation_failure"},
    )
    g.add_conditional_edges(
        "fast_executor",
        after_fast_execution,
        {"success": "answer_formatter", "retry": "sql_generator", "failed": "fast_execution_failure"},
    )
    g.add_edge("fast_validation_failure", END)
    g.add_edge("fast_execution_failure", END)

    # ================================================================
    # Complex path
    # ================================================================
    g.add_edge("query_decomposer", "sub_query_processor")
    g.add_edge("sub_query_processor", "sql_assembler")
    g.add_edge("sql_assembler", "complex_sql_validator")
    g.add_conditional_edges(
        "complex_sql_validator",
        route_complex_validation,
        {
            "continue": "complex_executor",
            "retry": "sub_query_processor",   # regenerate sub-queries with error context
            "failed": "complex_validation_failure",
        },
    )
    g.add_conditional_edges(
        "complex_executor",
        after_complex_execution,
        {
            "success": "answer_formatter",
            "retry": "sub_query_processor",
            "failed": "complex_execution_failure",
        },
    )
    g.add_edge("complex_validation_failure", END)
    g.add_edge("complex_execution_failure", END)

    # ================================================================
    # Shared finish
    # ================================================================
    g.add_edge("answer_formatter", "memory_update")
    g.add_edge("memory_update", "update_cache")
    g.add_edge("update_cache", END)

    return g.compile()
