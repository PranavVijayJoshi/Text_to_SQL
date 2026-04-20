"""
Rule-based guardrail. No LLM call — zero cost, zero false rejections.

Philosophy: default to ALLOW. Only reject when the query is unambiguously
non-database (greetings, opinions, coding help, harmful writes).
When in doubt, let it through — the planner will handle it.
"""
import re
from state import AgentState


# Reject only if the query clearly matches one of these categories.
# Patterns are intentionally narrow to avoid false positives.
_REJECT_PATTERNS = [
    # Greetings / small talk
    r"^(hi|hello|hey|howdy|good\s+(morning|afternoon|evening)|what'?s\s+up)[^\w]*$",

    # Explicit coding requests
    r"\b(write\s+(me\s+)?(a\s+)?(python|java|javascript|code|script|function|class|program))\b",
    r"\b(debug|refactor|explain\s+this\s+code)\b",

    # Opinion / general knowledge
    r"\b(what\s+do\s+you\s+think|your\s+opinion|recommend\s+me\s+a\s+(movie|book|song|restaurant))\b",
    r"\b(who\s+is\s+(the\s+)?(president|prime\s+minister|ceo\s+of\s+(?!our|the\s+company)))\b",
    r"\b(weather|news|stock\s+price\s+of\s+(?!our|the))\b",

    # Harmful write operations — explicit destructive intent
    r"\b(drop\s+(all\s+)?(tables?|database|schema))\b",
    r"\b(delete\s+(all|every)\s+(records?|rows?|data|customers?|orders?))\b",
    r"\b(truncate\s+(all|every|the)\s+\w+)\b",
]

_REJECT_RE = [re.compile(p, re.IGNORECASE) for p in _REJECT_PATTERNS]


def _is_rejected(query: str) -> tuple[bool, str]:
    q = query.strip()
    for pattern in _REJECT_RE:
        if pattern.search(q):
            return True, f"Query matched non-database pattern: {pattern.pattern}"
    return False, ""


def guardrail_analyzer(state: AgentState) -> AgentState:
    rejected, reason = _is_rejected(state["user_query"])
    return {
        **state,
        "is_relevant": not rejected,
        "rejection_reason": reason,
    }
