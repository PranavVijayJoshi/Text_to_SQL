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


def is_readonly_sql(sql: str) -> tuple[bool, list[str]]:
    if not sql.strip():
        return False, ["No SQL was generated."]

    if sqlglot is None or exp is None:
        return False, ["Missing dependency: install sqlglot to validate read-only SQL safely."]

    try:
        statements = sqlglot.parse(sql, read="postgres")
    except Exception as exc:
        return False, [f"SQL parse error: {exc}"]

    if len(statements) != 1:
        return False, ["Only one SQL statement is allowed."]

    tree = statements[0]
    read_roots = _expression_types(("Select", "Union", "Except", "Intersect"))
    if read_roots and not isinstance(tree, read_roots):
        return False, ["Only read-only SELECT queries are allowed."]

    forbidden_nodes = _expression_types(FORBIDDEN_EXPRESSION_NAMES)
    if forbidden_nodes and any(tree.find_all(*forbidden_nodes)):
        return False, ["Write, DDL, or command operation detected."]

    return True, []


def readonly_enforcer(state: AgentState) -> AgentState:
    sql = state.get("generated_sql", "")
    is_valid, errors = is_readonly_sql(sql)

    if is_valid:
        return {**state, "is_valid": True, "validation_errors": []}

    return {
        **state,
        "is_valid": False,
        "validation_errors": errors,
        "retry_count": state.get("retry_count", 0) + 1,
        "final_answer": (
            "Query rejected for safety reasons. "
            "This system only allows one read-only SELECT query."
        ),
        "status": "retrying",
        "error_message": "; ".join(errors),
    }
