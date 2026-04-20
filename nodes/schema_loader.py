from sqlalchemy import create_engine, inspect, text as sa_text
from state import AgentState


SAMPLE_TYPE_MARKERS = ("char", "text", "string", "enum", "citext", "user-defined", "bool")
MAX_SAMPLE_VALUES = 20


def _should_sample(column_type: str) -> bool:
    lowered = column_type.lower()
    return any(marker in lowered for marker in SAMPLE_TYPE_MARKERS)


def _sample_values(conn, preparer, table: str, column: str) -> list:
    quoted_table = preparer.quote(table)
    quoted_column = preparer.quote(column)
    sql = sa_text(
        f"SELECT DISTINCT {quoted_column} "
        f"FROM {quoted_table} "
        f"WHERE {quoted_column} IS NOT NULL "
        f"LIMIT {MAX_SAMPLE_VALUES}"
    )
    try:
        result = conn.execute(sql)
        return [row[0] for row in result.fetchall()]
    except Exception:
        return []


def schema_loader(state: AgentState) -> AgentState:
    """
    Load detailed context only for planned relevant tables.

    Includes columns, primary keys, foreign keys, and small samples for
    category-like columns so the planner/generator can avoid invalid values.
    """
    engine = create_engine(state["db_connection_string"])
    inspector = inspect(engine)
    preparer = engine.dialect.identifier_preparer

    schema = {}
    with engine.connect() as conn:
        for table in state["relevant_tables"]:
            cols = inspector.get_columns(table)
            pk_columns = set(inspector.get_pk_constraint(table).get("constrained_columns", []))

            foreign_keys = {}
            for fk in inspector.get_foreign_keys(table):
                referred_table = fk.get("referred_table")
                referred_columns = fk.get("referred_columns") or []
                for column, referred_column in zip(fk.get("constrained_columns") or [], referred_columns):
                    foreign_keys[column] = f"{referred_table}.{referred_column}"

            schema[table] = []
            for col in cols:
                column_type = str(col["type"])
                sample_values = (
                    _sample_values(conn, preparer, table, col["name"])
                    if _should_sample(column_type)
                    else []
                )
                schema[table].append(
                    {
                        "name": col["name"],
                        "type": column_type,
                        "primary_key": col["name"] in pk_columns,
                        "foreign_key": foreign_keys.get(col["name"]),
                        "nullable": col.get("nullable"),
                        "sample_values": sample_values,
                    }
                )

    return {**state, "schema": schema}
