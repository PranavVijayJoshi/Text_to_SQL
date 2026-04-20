from sqlalchemy import create_engine, text as sa_text
from state import AgentState


MAX_ROWS = 100
POSTGRES_STATEMENT_TIMEOUT = "15s"


def query_executor(state: AgentState) -> AgentState:
    try:
        engine = create_engine(state["db_connection_string"])
        with engine.begin() as conn:
            if engine.dialect.name == "postgresql":
                conn.execute(sa_text("SET TRANSACTION READ ONLY"))
                conn.execute(sa_text(f"SET LOCAL statement_timeout = '{POSTGRES_STATEMENT_TIMEOUT}'"))
            result = conn.execute(sa_text(state["generated_sql"]))
            rows = [dict(row._mapping) for row in result.fetchmany(MAX_ROWS)]

        final_answer = (
            f"Query executed successfully. Returned {len(rows)}"
            f"{' or more' if len(rows) == MAX_ROWS else ''} row(s)."
        )

        history = list(state.get("conversation_history", []))
        history.append(
            {
                "query": state["user_query"],
                "summary": f"Ran SQL: {state['generated_sql']}. Returned {len(rows)} rows.",
                "sql": state["generated_sql"],
            }
        )

        return {
            **state,
            "results": rows,
            "execution_error": None,
            "error_message": None,
            "validation_errors": [],
            "final_answer": final_answer,
            "status": "success",
            "conversation_history": history,
        }

    except Exception as exc:
        error = str(exc)
        return {
            **state,
            "results": [],
            "execution_error": error,
            "error_message": error,
            "validation_errors": [f"Execution error: {error}"],
            "retry_count": state.get("retry_count", 0) + 1,
            "final_answer": f"Execution failed: {error}",
            "status": "retrying",
        }
