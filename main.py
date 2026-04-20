import time

from config import DB_CONNECTION_STRING
from graph import build_graph
from state import initial_state


app = build_graph()


def run(user_query: str, user_id: str = "default_user", conversation_history=None):
    conversation_history = conversation_history or []
    start_state = initial_state(
        user_query=user_query,
        db_connection_string=DB_CONNECTION_STRING,
        user_id=user_id,
    )
    start_state["conversation_history"] = conversation_history

    final_state = app.invoke(start_state)
    return final_state["final_answer"], final_state.get("conversation_history", [])


if __name__ == "__main__":
    start = time.time()

    answer1, history = run(
        user_query="How many customers are there in the DB",
        user_id="user_128",
    )
    print(answer1)
    print("---")

    answer2, history = run(
        user_query="Just tell those names, who are more than 30 years of age.",
        user_id="user_128",
        conversation_history=history,
    )
    end = time.time()

    if not answer2:
        print("no answer 2")
    print(answer2)
    print(f"time taken: {end-start:.2f} seconds")
