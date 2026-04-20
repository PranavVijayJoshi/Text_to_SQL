import json

from langchain_core.messages import HumanMessage, SystemMessage
from state import AgentState
from llm import llm


def _fallback_answer(state: AgentState) -> str:
    rows = state.get("results", [])
    if not rows:
        return "I ran the query successfully, but it did not return any rows."

    if len(rows) == 1:
        items = ", ".join(f"{key}: {value}" for key, value in rows[0].items())
        return f"The query returned one row: {items}."

    return f"The query returned {len(rows)} rows. You can review the table below."


def answer_formatter(state: AgentState) -> AgentState:
    rows = state.get("results", [])
    rows_preview = json.dumps(rows[:20], indent=2, default=str)

    try:
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You convert SQL query results into a concise, human-readable answer.\n"
                        "Use the returned rows only. Do not invent facts.\n"
                        "If the rows are empty, say that no matching records were found.\n"
                        "Do not include the SQL query in the prose answer because it is shown separately."
                    )
                ),
                HumanMessage(
                    content=(
                        f"User question:\n{state['user_query']}\n\n"
                        f"SQL executed:\n{state['generated_sql']}\n\n"
                        f"Rows returned: {len(rows)}\n"
                        f"Rows preview:\n{rows_preview}"
                    )
                ),
            ]
        )
        answer = response.content.strip() or _fallback_answer(state)
    except Exception as exc:
        print(f"[answer_formatter error]: {exc}")
        answer = _fallback_answer(state)

    history = list(state.get("conversation_history", []))
    if history:
        history[-1] = {
            **history[-1],
            "summary": answer,
        }

    return {
        **state,
        "final_answer": answer,
        "status": "success",
        "error_message": None,
        "conversation_history": history,
    }
