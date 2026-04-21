# 🧠 Intelligent Natural Language to SQL Generator

A production-grade, LangGraph-powered agent that converts natural language questions into accurate SQL queries for any relational database. Built to handle queries of any complexity — from simple lookups to multi-level aggregations, recursive hierarchies, and anti-join patterns.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Running the System](#running-the-system)
- [API Reference](#api-reference)
- [Example Queries](#example-queries)
- [How It Works](#how-it-works)
- [Design Decisions](#design-decisions)

---

## Overview

Most natural language to SQL systems fail on complex analytical queries — multi-level aggregations, HAVING clauses, recursive hierarchies, anti-joins, and date arithmetic. This system solves that by **routing queries by complexity** rather than sending everything through a single LLM pipeline.

Simple queries go through a fast, cheap path. Complex queries are automatically decomposed into smaller, independently solvable sub-queries, each handled by a focused LLM call, then assembled into a single optimised SQL query using CTEs.

**Built on a real e-commerce database schema** (customers, orders, order items, products, categories) with full hierarchical category support.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│           INTAKE LAYER              │
│  Metadata Loader → Guardrail →      │
│  Memory Retrieval → Table Selector  │
│  → Schema Loader                    │
└─────────────────┬───────────────────┘
                  │
                  ▼
        ┌─────────────────┐
        │   COMPLEXITY    │
        │   CLASSIFIER    │
        │  (Rule-based,   │
        │   zero LLM cost)│
        └────────┬────────┘
                 │
        ┌────────┴────────┐
        │                 │
   Tier 1/2          Tier 3/4
   (Fast Path)    (Complex Path)
        │                 │
        ▼                 ▼
  ┌──────────┐    ┌───────────────┐
  │ Query    │    │ Query         │
  │ Planner  │    │ Decomposer    │
  │          │    │ (Gemini/70B)  │
  └────┬─────┘    └──────┬────────┘
       │                 │
       ▼                 ▼
  ┌──────────┐    ┌───────────────┐
  │   Plan   │    │  Sub-Query    │
  │ Validator│    │  Processor    │
  └────┬─────┘    └──────┬────────┘
       │                 │
       ▼                 ▼
  ┌──────────┐    ┌───────────────┐
  │   SQL    │    │  SQL Assembler│
  │ Generator│    │  (CTE builder)│
  └────┬─────┘    └──────┬────────┘
       │                 │
       └────────┬─────────┘
                │
                ▼
        ┌───────────────┐
        │ SQL Validator │
        │ (sqlglot +    │
        │  DB EXPLAIN)  │
        └──────┬────────┘
               │
               ▼
        ┌───────────────┐
        │ Query Executor│
        └──────┬────────┘
               │
               ▼
        ┌───────────────┐
        │    Answer     │
        │   Formatter   │
        └──────┬────────┘
               │
               ▼
        ┌───────────────┐
        │ Memory Update │
        │ + Cache Update│
        └───────────────┘
```

---

## Key Features

### 🔀 Complexity-Based Routing
Queries are classified into 4 tiers using rule-based heuristics (zero LLM cost):
- **Tier 1** — Single table, simple filters (`Show customers from Mumbai`)
- **Tier 2** — Multi-table join with basic aggregation (`Total revenue by category`)
- **Tier 3** — Window functions, HAVING, anti-joins, date arithmetic, recursive hierarchy (`Products never ordered`, `Top category per customer`, `All sub-categories combined`)
- **Tier 4** — Multi-level aggregation with filtering on aggregated results (`Customers with 3+ orders above average spend`)

### 🧩 Query Decomposition (Tier 3/4)
Complex queries are broken into a DAG of atomic sub-queries. Each node is simple enough for an 8B model to handle reliably. The decomposer uses a stronger model (70B/Gemini) once, then each sub-query is handled cheaply.

### 🏗️ CTE Assembly
Sub-query SQL fragments are assembled deterministically into a single `WITH` (CTE) query. No LLM involved in assembly — fully rule-based and reliable.

### 💾 Three-Layer Memory
- **Semantic memory** — Reusable facts about the database schema extracted from successful queries
- **Episodic memory** — Per-user query history for conversation context
- **Procedural memory** — Learned rules from past failures to prevent repeat errors

### ⚡ Query Caching
Exact-match caching with TTL. Repeated queries return instantly without any LLM calls.

### 🔒 Safety
- Rule-based guardrail — rejects non-database queries without an LLM call
- Read-only enforcement via sqlglot AST parsing
- SQL validated by sqlglot parser + PostgreSQL `EXPLAIN` before execution
- Statement timeout on all database queries

### 🌐 Cross-Database Support
The plan schema is dialect-neutral. sqlglot handles transpilation to PostgreSQL, MySQL, SQLite, SQL Server, and more.

---

## Tech Stack

| Component | Technology |
|---|---|
| Agent Framework | LangGraph |
| LLM (Fast) | Groq — `llama-3.1-8b-instant` |
| LLM (Strong) | Groq — `llama-3.3-70b-versatile` |
| SQL Parsing & Validation | sqlglot |
| Database | PostgreSQL |
| ORM / DB Connector | SQLAlchemy |
| Memory Store | LangMem (InMemoryStore) |
| Embeddings | Google Gemini Embeddings |
| API Layer | FastAPI + Uvicorn |
| Environment | Python 3.11+ |

---

## Project Structure

```
SQL_generator_langraph/
│
├── api.py                      # FastAPI app — REST endpoint
├── main.py                     # CLI runner for direct testing
├── graph.py                    # LangGraph graph definition and routing
├── state.py                    # AgentState TypedDict
├── llm.py                      # Model configuration (llm + llm_strong)
├── config.py                   # DB connection string, constants
├── memory_store.py             # LangMem store initialisation
├── requirements.txt
├── .env.example
├── schema.sql                  # Database schema + seed data
│
└── nodes/
    ├── metadata_loader.py          # Loads table names and comments
    ├── guardrail_analyzer.py       # Rule-based query relevance filter
    ├── memory_retrieval.py         # Retrieves semantic/episodic/procedural memory
    ├── table_selector.py           # Identifies relevant tables for the query
    ├── schema_loader.py            # Loads detailed schema for selected tables
    ├── complexity_classifier.py    # Routes query to fast or complex path
    │
    │   ── Fast Path (Tier 1/2) ──
    ├── query_planner.py            # Produces structured JSON query plan
    ├── plan_validator.py           # Validates plan against schema
    ├── sql_generator.py            # Generates SQL from plan
    │
    │   ── Complex Path (Tier 3/4) ──
    ├── query_decomposer.py         # Breaks query into DAG of sub-intents
    ├── sub_query_processor.py      # Generates SQL for each DAG node
    ├── sql_assembler.py            # Assembles CTEs into final SQL
    │
    │   ── Shared ──
    ├── sql_validator.py            # sqlglot parse + DB EXPLAIN validation
    ├── semantic_sql_validator.py   # Checks SQL matches the plan intent
    ├── query_executor.py           # Executes SQL read-only with timeout
    ├── answer_formatter.py         # Converts rows to natural language answer
    ├── query_cache.py              # In-memory query result cache
    ├── memory_update.py            # Extracts and stores memory after success
    └── readonly_enforcer.py        # AST-level write operation guard
```

---

## Setup & Installation

### Prerequisites
- Python 3.11+
- PostgreSQL running locally
- A free [Groq API key](https://console.groq.com)

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment variables
```bash
cp .env.example .env
```
Edit `.env` and fill in your values:
```
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key   # only needed if using Gemini for llm_strong
DB_USER=postgres
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=sql_generator
```

### 4. Set up the database
```bash
psql -U postgres -c "CREATE DATABASE sql_generator;"
psql -U postgres -d sql_generator -f schema.sql
```

---

## Running the System

### Option A — FastAPI server
```bash
uvicorn api:app --host localhost --port 8000 --reload
```
Visit `http://localhost:8000/docs` for the interactive Swagger UI.

### Option B — CLI
```bash
python main.py
```

---

## API Reference

### `POST /query`

**Request:**
```json
{
  "user_query": "What is the total revenue generated by each city?",
  "user_id": "user_123",
  "conversation_history": []
}
```

**Response:**
```json
{
  "answer": "The total revenue by city is: Pune — ₹4,033, Mumbai — ₹1,499...",
  "status": "success",
  "sql": "WITH city_revenue AS (...) SELECT ...",
  "rows": [...],
  "rows_returned": 5,
  "time_taken_seconds": 3.42,
  "cache_hit": false,
  "error_message": null,
  "conversation_history": [...]
}
```

### `GET /health`
Returns `{"status": "ok"}` — use for uptime checks.

### `DELETE /cache`
Clears the query cache. Call this after schema changes.

---

## Example Queries

| Query | Tier | Path |
|---|---|---|
| Show emails of customers from Mumbai | 1 | Fast |
| How many products has Pranav Joshi purchased? | 2 | Fast |
| What is the total revenue by category? | 2 | Fast |
| Which product category has the highest average unit price? | 3 | Complex |
| List customers who placed more than 2 orders | 3 | Complex |
| Which products have never been ordered? | 3 | Complex |
| Find orders placed in the last 15 days of April 2024 | 3 | Complex |
| Total sales for Electronics including all sub-categories | 3 | Complex |
| For each city, find the customer who spent the most | 3 | Complex |
| Find customers from Pune with 3+ orders, above average spend, show top category | 4 | Complex |

---

## How It Works

### Fast Path (Tier 1/2)
1. Schema for relevant tables is loaded
2. LLM produces a structured JSON plan (metrics, dimensions, filters, joins)
3. Plan is validated against the schema
4. SQL is generated from the validated plan
5. SQL is parsed by sqlglot and verified by PostgreSQL `EXPLAIN`
6. Query executes and result is formatted into natural language

### Complex Path (Tier 3/4)
1. Schema for relevant tables is loaded
2. A stronger LLM decomposes the query into an ordered DAG of 3–5 atomic sub-intents
3. Each node in the DAG generates its own SQL independently (simple enough for 8B model)
4. SQL Assembler combines all fragments into a single `WITH` CTE query deterministically
5. Same validation and execution pipeline as fast path

### Memory System
After every successful query, the system extracts:
- **Semantic facts** — schema insights reusable across all users (e.g. "revenue is stored in order_items.unit_price")
- **Episodes** — the query, SQL, and outcome stored per user for conversation continuity
- **Procedural rules** — failure lessons stored per user to prevent repeat errors

---

## Design Decisions

**Why rule-based complexity classification?**
An LLM-based classifier would add latency and cost to every query. Pattern matching is instant, free, and deterministic. Errors in classification are debuggable by reading a list of regex patterns — not by prompting a black box.

**Why decompose instead of prompting harder?**
A single LLM call producing a complete plan for a 4-table analytical query with nested aggregation is unreliable even for large models. Decomposition turns one hard problem into 4–5 trivial ones. Reliability improves dramatically.

**Why CTEs instead of nested subqueries?**
CTEs are independently readable, testable, and debuggable. The assembler can build them deterministically from the DAG. Nested subqueries require the model to reason about scope and aliasing — a common source of errors.

**Why sqlglot for validation?**
sqlglot parses SQL structurally — it understands CTEs, aliases, and nested queries correctly. Regex-based SQL validation always has edge cases. Running `EXPLAIN` on the database catches semantic errors (wrong column names, type mismatches) that static parsing misses.

**Why a rule-based guardrail?**
An LLM guardrail was incorrectly rejecting valid database queries it found "complex" or "ambiguous". Rule-based rejection with a default-allow policy eliminates false rejections entirely. The guardrail's only job is blocking obviously non-database requests — not judging query difficulty.
