from state import AgentState

try:
    import sqlglot
    from sqlglot import exp
except ImportError:  # pragma: no cover - depends on local environment
    sqlglot = None
    exp = None


FORBIDDEN_EXPRESSION_NAMES = (
    "Insert",
    "Update",
    "Delete",
    "Drop",
    "Create",
    "Alter",
    "Truncate",
    "Command",
)


def _expression_types(names: tuple[str, ...]) -> tuple[type, ...]:
    if exp is None:
        return ()
    return tuple(getattr(exp, name) for name in names if hasattr(exp, name))


def _read_query_roots() -> tuple[type, ...]:
    return _expression_types(("Select", "Union", "Except", "Intersect"))


def _forbidden_expression_types() -> tuple[type, ...]:
    return _expression_types(FORBIDDEN_EXPRESSION_NAMES)


def _cte_names(tree) -> set[str]:
    return {
        cte.alias_or_name.lower()
        for cte in tree.find_all(exp.CTE)
        if cte.alias_or_name
    }


def _referenced_db_tables(tree) -> set[str]:
    cte_names = _cte_names(tree)
    referenced_tables = set()

    for table in tree.find_all(exp.Table):
        table_name = table.name
        if not table_name:
            continue

        table_name = table_name.lower()
        if table_name in cte_names:
            continue

        referenced_tables.add(table_name)

    return referenced_tables


def _database_plan_errors(sql: str, connection_string: str) -> list[str]:
    """
    Ask the database planner to validate names/types without running the query.

    This catches issues static checks miss, such as referencing
    order_items.category_id when category_id actually lives on products.
    """
    from sqlalchemy import create_engine, text as sa_text

    sql_to_explain = sql.strip().rstrip(";")
    if not sql_to_explain:
        return ["No SQL was generated."]

    try:
        engine = create_engine(connection_string)
        with engine.begin() as conn:
            if engine.dialect.name == "postgresql":
                conn.execute(sa_text("SET TRANSACTION READ ONLY"))
                conn.execute(sa_text("SET LOCAL statement_timeout = '10s'"))
            conn.execute(sa_text(f"EXPLAIN {sql_to_explain}"))
    except Exception as exc:
        details = "\n".join(str(exc).splitlines()[:6])
        return [f"Database planner rejected SQL before execution: {details}"]

    return []


def sql_validator(state: AgentState) -> AgentState:
    """
    Validate generated SQL without executing it.

    This uses sqlglot instead of regex so CTE names, table aliases, nested
    queries, and schema-qualified tables are handled structurally.
    """
    sql = state.get("generated_sql", "")
    errors: list[str] = []

    if not sql.strip():
        errors.append("No SQL was generated.")
    elif sqlglot is None or exp is None:
        errors.append("Missing dependency: install sqlglot to validate SQL safely.")
    else:
        statements = []
        try:
            statements = sqlglot.parse(sql, read="postgres")
        except Exception as exc:
            errors.append(f"SQL parse error: {exc}")

        if len(statements) != 1:
            errors.append("Only one SQL statement is allowed.")

        if statements:
            tree = statements[0]
            read_roots = _read_query_roots()
            if read_roots and not isinstance(tree, read_roots):
                errors.append("Only read-only SELECT queries are allowed.")

            forbidden_types = _forbidden_expression_types()
            if forbidden_types and any(tree.find_all(*forbidden_types)):
                errors.append("Write, DDL, or command operation detected.")

            schema_tables = {table.lower() for table in state["schema"].keys()}
            for table in sorted(_referenced_db_tables(tree)):
                if table not in schema_tables:
                    errors.append(f"Table '{table}' not found in schema.")

        if not errors:
            errors.extend(
                _database_plan_errors(
                    sql=sql,
                    connection_string=state["db_connection_string"],
                )
            )

    is_valid = len(errors) == 0
    retry_count = state.get("retry_count", 0) + (0 if is_valid else 1)

    return {
        **state,
        "is_valid": is_valid,
        "validation_errors": errors,
        "retry_count": retry_count,
    }
