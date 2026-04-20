from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from typing import Any
import uvicorn
import time
from graph import build_graph
from config import DB_CONNECTION_STRING
from state import initial_state


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ConversationTurn(BaseModel):
    query: str
    summary: str
    sql: str | None = None


class QueryRequest(BaseModel):
    user_query: str
    user_id: str = "default_user"
    conversation_history: list[ConversationTurn] = Field(default_factory=list)


class QueryResponse(BaseModel):
    answer: str
    status: str
    sql: str
    rows: list[dict[str, Any]]
    rows_returned: int
    time_taken_seconds: float
    cache_hit: bool
    error_message: str | None = None
    conversation_history: list[dict]


# ---------------------------------------------------------------------------
# App lifecycle — build graph once at startup
# ---------------------------------------------------------------------------

app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    app_state["graph"] = build_graph()
    print("LangGraph compiled and ready.")
    yield
    app_state.clear()

app = FastAPI(
    title="SQL Generator Agent",
    description="Natural language to SQL agent with memory",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    start = time.time()

    try:
        graph       = app_state["graph"]
        start_state = initial_state(
            user_query=request.user_query,
            db_connection_string=DB_CONNECTION_STRING,
            user_id=request.user_id,
        )
        start_state["conversation_history"] = [
            turn.model_dump() if hasattr(turn, "model_dump") else turn.dict()
            for turn in request.conversation_history
        ]

        final_state = graph.invoke(start_state)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    rows = jsonable_encoder(final_state.get("results", []))

    return QueryResponse(
        answer=final_state["final_answer"],
        status=final_state.get("status", "success" if not final_state.get("execution_error") else "failed"),
        sql=final_state.get("generated_sql", ""),
        rows=rows,
        rows_returned=len(rows),
        time_taken_seconds=round(time.time() - start, 2),
        cache_hit=final_state.get("cache_hit", False),
        error_message=final_state.get("error_message") or final_state.get("execution_error"),
        conversation_history=final_state.get("conversation_history", []),
    )


@app.delete("/cache")
def clear_cache():
    """Clear the query cache — useful after schema changes."""
    from nodes.query_cache import _cache
    _cache.clear()
    return {"status": "cache cleared"}


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    #print("hi")
    uvicorn.run("api:app", host="localhost", port=8000, reload=True)
