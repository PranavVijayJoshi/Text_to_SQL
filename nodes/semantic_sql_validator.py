import re

from state import AgentState

try:
    import sqlglot
    from sqlglot import exp
except ImportError:  # pragma: no cover - depends on local environment
    sqlglot = None
    exp = None


def _column_name(field: str) -> str:
    return field.split(".")[-1].strip().lower()


def _normalise_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.lower())


def _has_grouping(sql: str, tree) -> bool:
    normalised = _normalise_sql(sql)
    if " group by " in normalised or " over " in normalised:
        return True
    if tree is not None and exp is not None:
        return any(tree.find_all(exp.Group))
    return False


def _parse_tree(sql: str):
    if sqlglot is None:
        return None
    try:
        statements = sqlglot.parse(sql, read="postgres")
    except Exception:
        return None
    return statements[0] if statements else None


def _value_appears_in_sql(value, normalised_sql: str) -> bool:
    value_text = str(value).lower()
    if not value_text:
        return True
    if value_text in normalised_sql:
        return True
    year_match = re.match(r"^(\d{4})-\d{2}-\d{2}$", value_text)
    return bool(year_match and year_match.group(1) in normalised_sql)


def _field_appears_in_sql(field, normalised_sql: str) -> bool:
    if not field or str(field) == "*":
        return True
    return _column_name(str(field)) in normalised_sql


def semantic_sql_validator(state: AgentState) -> AgentState:
    plan = state.get("query_plan", {})
    sql = state.get("generated_sql", "")
    normalised = _normalise_sql(sql)
    tree = _parse_tree(sql)
    errors = []

    for filter_item in plan.get("filters", []):
        if not isinstance(filter_item, dict):
            continue

        field = filter_item.get("field")
        value = filter_item.get("value")
        if field and not _field_appears_in_sql(field, normalised):
            errors.append(f"SQL appears to be missing planned filter field: {field}")

        if value is not None:
            values = value if isinstance(value, list) else [value]
            for item in values:
                item_text = str(item).lower()
                if item_text and not _value_appears_in_sql(item, normalised):
                    errors.append(f"SQL appears to be missing planned filter value: {item}")

    time_scope = plan.get("time_scope")
    if isinstance(time_scope, dict):
        field = time_scope.get("field")
        if field and not _field_appears_in_sql(field, normalised):
            errors.append(f"SQL appears to be missing planned time field: {field}")
        for key in ("start", "end"):
            value = time_scope.get(key)
            if value and not _value_appears_in_sql(value, normalised):
                errors.append(f"SQL appears to be missing planned time {key}: {value}")

    aggregation_scope = str(plan.get("aggregation_scope", "")).lower()
    metrics = plan.get("metrics") or []
    dimensions = plan.get("dimensions") or []
    if aggregation_scope.startswith("per_") and metrics and dimensions and not _has_grouping(sql, tree):
        errors.append(
            f"Plan requires grouped/windowed aggregation for scope '{aggregation_scope}', "
            "but SQL does not appear to group or use a window."
        )

    for dimension in dimensions:
        field = dimension.get("field") if isinstance(dimension, dict) else dimension
        if field and not _field_appears_in_sql(field, normalised):
            errors.append(f"SQL appears to be missing planned dimension: {field}")

    for metric in metrics:
        if not isinstance(metric, dict):
            continue
        field = metric.get("field")
        if field and not _field_appears_in_sql(field, normalised):
            errors.append(f"SQL appears to be missing planned metric field: {field}")

    for join in plan.get("joins", []):
        if not isinstance(join, dict):
            continue
        for side in ("left", "right"):
            field = join.get(side)
            if field and not _field_appears_in_sql(field, normalised):
                errors.append(f"SQL appears to be missing planned join field: {field}")

    for output_item in plan.get("output", []):
        field = output_item.get("field") if isinstance(output_item, dict) else output_item
        if isinstance(field, str) and "." in field and not _field_appears_in_sql(field, normalised):
            errors.append(f"SQL appears to be missing planned output field: {field}")

    ranking = plan.get("ranking")
    if isinstance(ranking, dict) and ranking:
        if ranking.get("order_by") and " order by " not in normalised:
            errors.append("Plan requires ranking/order_by, but SQL has no ORDER BY.")
        if ranking.get("limit") is not None and " limit " not in normalised:
            errors.append("Plan requires a ranking limit, but SQL has no LIMIT.")

    hierarchy = plan.get("hierarchy")
    if isinstance(hierarchy, dict) and hierarchy.get("required"):
        if "with recursive" not in normalised:
            errors.append("Plan requires hierarchical traversal, but SQL has no WITH RECURSIVE.")

    is_valid = len(errors) == 0
    retry_count = state.get("retry_count", 0) + (0 if is_valid else 1)

    return {
        **state,
        "is_valid": is_valid,
        "semantic_validation_errors": errors,
        "validation_errors": errors,
        "retry_count": retry_count,
    }
