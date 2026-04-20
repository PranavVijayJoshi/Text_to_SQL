from typing import Any, TypedDict


class AgentState(TypedDict):
    # Inputs
    user_query: str
    db_connection_string: str
    user_id: str
    sql_dialect: str           # "postgres" | "mysql" | "sqlite" | "tsql" etc.

    # Guardrail
    is_relevant: bool
    rejection_reason: str

    # Metadata and schema context
    table_names: list[str]
    table_metadata: dict[str, str]
    relevant_tables: list[str]
    schema: dict[str, list[dict]]

    # Complexity routing
    complexity_tier: int        # 1=simple, 2=join+agg, 3=window/having, 4=multi-level

    # Fast path (tier 1/2) — structured reasoning
    query_intent: str
    query_plan: dict[str, Any]
    plan_errors: list[str]

    # Complex path (tier 3/4) — decomposition DAG
    # Each node: {id, intent, tables, depends_on, sql}
    sub_queries: list[dict]

    # SQL generation and validation (shared)
    generated_sql: str
    is_valid: bool
    validation_errors: list[str]
    semantic_validation_errors: list[str]
    retry_count: int

    # Execution
    results: list[dict[str, Any]]
    execution_error: str | None

    # Final output
    final_answer: str
    status: str
    error_message: str | None

    # Memory
    semantic_facts: list[str]
    past_episodes: list[str]
    procedural_rules: list[str]
    conversation_history: list[dict]

    # Cache
    cache_hit: bool


def initial_state(
    user_query: str,
    db_connection_string: str,
    user_id: str = "user_123",
    sql_dialect: str = "postgres",
) -> AgentState:
    return AgentState(
        user_query=user_query,
        db_connection_string=db_connection_string,
        user_id=user_id,
        sql_dialect=sql_dialect,
        is_relevant=True,
        rejection_reason="",
        table_names=[],
        table_metadata={},
        relevant_tables=[],
        schema={},
        complexity_tier=1,
        query_intent="",
        query_plan={},
        plan_errors=[],
        sub_queries=[],
        generated_sql="",
        is_valid=False,
        validation_errors=[],
        semantic_validation_errors=[],
        retry_count=0,
        results=[],
        execution_error=None,
        final_answer="",
        status="pending",
        error_message=None,
        semantic_facts=[],
        past_episodes=[],
        procedural_rules=[],
        conversation_history=[],
        cache_hit=False,
    )
