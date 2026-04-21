"""
Classifies the user query into a complexity tier using rule-based heuristics.
No LLM call — zero cost, instant.

Tier 1: Single table, simple filter / lookup
Tier 2: Multi-table join, one level of GROUP BY
Tier 3: Window functions, HAVING on aggregation, top-N-per-group
Tier 4: Multi-level aggregation — filter on an aggregated result
         (e.g. "above average spend", "at least N orders" + secondary metric)

Tiers 1-2 → fast path (existing planner + SQL generator)
Tiers 3-4 → complex path (decomposer + sub-query processor + assembler)
"""
import re
from state import AgentState


# ------------------------------------------------------------------
# Pattern sets — each is checked case-insensitively
# ------------------------------------------------------------------

# Strong signals for tier 4: filtering ON an aggregated result
_TIER4_PATTERNS = [
    r"\babove\s+(the\s+)?average\b",
    r"\bgreater\s+than\s+(the\s+)?average\b",
    r"\bhigher\s+than\s+(the\s+)?average\b",
    r"\bmore\s+than\s+(the\s+)?average\b",
    r"\bexceed(s|ing)?\s+(the\s+)?average\b",
    r"\bat\s+least\s+\d+\s+orders?\b",
    r"\bat\s+least\s+\d+\s+purchases?\b",
    r"\bat\s+least\s+\d+\s+times?\b",
    r"\bno\s+fewer\s+than\s+\d+\b",
    # Sequential event pattern — requires two passes over the same table
    r"\bfirst\s+(purchase|order).{1,60}(second|next)\s+(purchase|order)\b",
    r"\b(second|next)\s+(purchase|order).{1,60}first\s+(purchase|order)\b",
    # Recursive hierarchy + aggregation — always multi-level
    r"\ball\s+(of\s+its\s+|its\s+|their\s+)?(sub-?categor\w*|subcategor\w*|children|descendants?)\b",
    r"\bsub-?categor\w*\s+(and|combined|together|included)\b",
    r"\band\s+all\s+(its\s+|their\s+)?sub-?categor\w*\b",
    r"\bentire\s+(hierarch\w*|tree|category\s+tree)\b",
    r"\bincluding\s+(all\s+)?(its\s+|their\s+)?(sub-?categor\w*|children|descendants?)\b",
    r"\brecurs\w*\b",
]

# Signals for tier 3: single complex aggregation
_TIER3_PATTERNS = [
    r"\btop\s+\d+\b",
    r"\bmost\s+(popular|frequent|common|expensive|purchased|bought)\b",
    r"\bbest[\s-]selling\b",
    r"\brank(ed|ing)?\b",
    r"\bwindow\s+function\b",
    r"\brunning\s+(total|sum|average|count)\b",
    r"\bcumulative\b",
    r"\bpercentile\b",
    r"\bmedian\b",
    r"\bntile\b",
    r"\bhighest.{1,30}(per|for\s+each)\b",
    r"\blowest.{1,30}(per|for\s+each)\b",

    # Aggregation ranking without "per" — e.g. "highest average unit price"
    r"\b(highest|lowest)\s+(average|avg|total|mean|sum)\b",
    r"\b(highest|lowest)\s+\w+\s+(price|revenue|amount|cost|value)\b",
    r"\bwhich\s+\w+\s+has\s+the\s+(highest|lowest|most|least)\b",

    # HAVING on count — e.g. "more than 2 orders", "over 3 purchases"
    r"\bmore\s+than\s+\d+\s+(orders?|purchases?|times?|items?|products?)\b",
    r"\bover\s+\d+\s+(orders?|purchases?|times?|items?|products?)\b",
    r"\bplaced\s+more\s+than\s+\d+\b",

    # Negation / anti-join — e.g. "never been ordered", "not purchased"
    r"\bnever\s+(been\s+)?(ordered|purchased|bought|sold|reviewed)\b",
    r"\bnot\s+(been\s+)?(ordered|purchased|bought|sold)\b",
    r"\bhave\s+no\s+orders?\b",
    r"\bwith\s+no\s+orders?\b",
    r"\bwithout\s+(any\s+)?(orders?|purchases?|reviews?)\b",
    r"\bzero\s+(orders?|purchases?|sales?)\b",

    # Date arithmetic — e.g. "last 15 days of April", "first N days"
    r"\blast\s+\d+\s+days?\b",
    r"\bfirst\s+\d+\s+days?\b",
    r"\bdays?\s+between\b",
    r"\bnumber\s+of\s+days?\b",
    r"\bdate\s+difference\b",
    r"\bdiff(erence)?\s+between.{1,30}(date|order|purchase)\b",

    # Per-location/group superlative — e.g. "for each city, who spent the most"
    r"\bfor\s+each\s+(city|region|country|state|store|location|department|branch)\b",
    r"\bwho\s+(has\s+)?(spent|bought|purchased|ordered)\s+the\s+most\b",
    r"\bmost\s+(money|revenue|amount|value|sales)\b",
    r"\b(highest|lowest)\s+(spend|spending|sales|revenue)\s+(per|for|by|in)\b",
]

# Signals for tier 2: any aggregation or multi-table concept
_TIER2_PATTERNS = [
    r"\btotal\b",
    r"\bcount\b",
    r"\bsum\b",
    r"\baverage\b",
    r"\bmean\b",
    r"\bper\s+(customer|product|category|month|year|day|region|store)\b",
    r"\bfor\s+each\s+(customer|product|category)\b",
    r"\bgroup\b",
    r"\bhow\s+many\b",
    r"\bnumber\s+of\b",
    r"\bamount\s+(spent|paid|ordered)\b",
]

# Multi-table presence signals
_MULTI_TABLE_WORDS = [
    "order", "purchase", "product", "category", "customer",
    "invoice", "item", "payment", "shipment", "review",
]


def _matches(query: str, patterns: list[str]) -> bool:
    return any(re.search(p, query, re.IGNORECASE) for p in patterns)


def _multi_table_count(query: str) -> int:
    q = query.lower()
    return sum(1 for w in _MULTI_TABLE_WORDS if w in q)


def classify(query: str) -> int:
    # Tier 4: filter ON an aggregated result (always needs decomposition)
    if _matches(query, _TIER4_PATTERNS):
        return 4

    # Tier 3: complex single-pass aggregation (ranking, top-N-per-group)
    if _matches(query, _TIER3_PATTERNS):
        return 3

    # Tier 2: simple aggregation or multi-table join
    if _matches(query, _TIER2_PATTERNS) or _multi_table_count(query) >= 2:
        return 2

    return 1


def complexity_classifier(state: AgentState) -> AgentState:
    tier = classify(state["user_query"])
    return {**state, "complexity_tier": tier}
