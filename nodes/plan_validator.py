"""
Validates the structured query plan produced by query_planner (tier 1/2 only).
Checks structural integrity — no hardcoded domain heuristics.
"""
from state import AgentState

MAX_SAMPLE_VALUES = 20


def _field_map(schema: dict) -> dict:
    fields = {}
    for table, columns in schema.items():
        for col in columns:
            name = col.get("name")
            if not name:
                continue
            fields[name.lower()] = col
            fields[f"{table.lower()}.{name.lower()}"] = col
    return fields


def _field_exists(field: str, field_map: dict) -> bool:
    if not field or field.strip() in ("*", ""):
        return True
    # Accept expressions like "unit_price * quantity"
    if any(op in field for op in ("*", "+", "-", "/")):
        return True
    return field.lower() in field_map


def _iter_fields(plan: dict):
    for m in plan.get("metrics", []):
        if isinstance(m, dict):
            yield m.get("field")
    for d in plan.get("dimensions", []):
        yield (d.get("field") if isinstance(d, dict) else d)
    for f in plan.get("filters", []):
        if isinstance(f, dict):
            yield f.get("field")
    for j in plan.get("joins", []):
        if isinstance(j, dict):
            yield j.get("left")
            yield j.get("right")
    ts = plan.get("time_scope")
    if isinstance(ts, dict):
        yield ts.get("field")


def _validate_filter_values(plan: dict, field_map: dict) -> list[str]:
    errors = []
    for f in plan.get("filters", []):
        if not isinstance(f, dict):
            continue
        field = f.get("field")
        value = f.get("value")
        col = field_map.get(field.lower()) if field else None
        if not col or value is None:
            continue
        samples = col.get("sample_values") or []
        if not samples or len(samples) >= MAX_SAMPLE_VALUES:
            continue
        valid = {str(s).lower() for s in samples}
        requested = value if isinstance(value, list) else [value]
        bad = [v for v in requested if str(v).lower() not in valid]
        if bad:
            errors.append(
                f"Filter value {bad} not known for '{field}'. Known: {samples}"
            )
    return errors


def plan_validator(state: AgentState) -> AgentState:
    plan = state.get("query_plan", {})
    schema = state.get("schema", {})
    errors = []

    if not plan:
        errors.append("No query plan was produced.")
    else:
        operational_keys = ("metrics", "dimensions", "filters", "joins", "ranking", "time_scope", "output")
        if not any(plan.get(k) for k in operational_keys):
            errors.append(
                "Plan has no operational details. "
                "It must include at least one of: metrics, dimensions, filters, joins, ranking, time_scope, or output."
            )

    schema_tables = {t.lower() for t in schema}
    planned_tables = {t.lower() for t in plan.get("tables", []) if isinstance(t, str)}
    missing = sorted(planned_tables - schema_tables)
    if missing:
        errors.append(f"Plan references tables not in schema: {missing}")

    if len(planned_tables) > 1 and not plan.get("joins"):
        errors.append("Plan uses multiple tables but defines no joins.")

    fm = _field_map(schema)
    for field in _iter_fields(plan):
        if field and not _field_exists(str(field), fm):
            errors.append(f"Plan references unknown field: {field}")

    errors.extend(_validate_filter_values(plan, fm))

    ranking = plan.get("ranking")
    if isinstance(ranking, dict) and ranking:
        if not ranking.get("order_by"):
            errors.append("Ranking requires an order_by field.")
        limit = ranking.get("limit")
        if limit is not None:
            try:
                if int(limit) <= 0:
                    errors.append("Ranking limit must be a positive integer.")
            except (TypeError, ValueError):
                errors.append("Ranking limit must be a number.")

    is_valid = len(errors) == 0
    retry_count = state.get("retry_count", 0) + (0 if is_valid else 1)

    return {
        **state,
        "is_valid": is_valid,
        "plan_errors": errors,
        "validation_errors": errors,
        "retry_count": retry_count,
    }
