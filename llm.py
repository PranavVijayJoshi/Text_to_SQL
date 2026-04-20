import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

# Rate limiter — stays within free tier limits
_groq_limiter = InMemoryRateLimiter(
    requests_per_second=0.25,
    check_every_n_seconds=0.5,
)
_gemini_limiter = InMemoryRateLimiter(
    requests_per_second=0.5,
    check_every_n_seconds=0.5,
)

# -----------------------------------------------------------------------
# llm  — Groq Llama 3.1 8B
# Used for: guardrail, table selection, per-node planning, SQL generation,
#           semantic validation, answer formatting
# Fast and cheap. Handles tier 1/2 queries and sub-query SQL generation.
# -----------------------------------------------------------------------



llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
    temperature=0,
    rate_limiter=_groq_limiter,
)

# llm = ChatOllama(
#     model="llama3.1:8b",
#     temperature=0,
# )
# -----------------------------------------------------------------------
# llm_strong  — Gemini 2.0 Flash
# Used ONLY for: query decomposition (one call per tier 3/4 query)
# Stronger reasoning needed to correctly break a complex query into a DAG.
# -----------------------------------------------------------------------
llm_strong = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
    temperature=0,
    rate_limiter=_groq_limiter,
)


# llm_strong = ChatOllama(
#     model="llama3.1:8b",
#     temperature=0,
# )