import re
from langchain_core.messages import HumanMessage, SystemMessage
from state import AgentState
from llm import llm


def metadata_loader(state: AgentState) -> AgentState:
    """
    Fetch only table names and optional comments — no column detail.
    Stays lightweight even with hundreds of tables (~500 tokens total).
    """
    from sqlalchemy import create_engine, inspect

    engine    = create_engine(state["db_connection_string"])
    inspector = inspect(engine)

    table_names    = inspector.get_table_names()
    table_metadata = {}

    for name in table_names:
        try:
            comment = inspector.get_table_comment(name).get("text", "")
        except Exception:
            comment = ""
        table_metadata[name] = comment

    return {**state, "table_names": table_names, "table_metadata": table_metadata}
