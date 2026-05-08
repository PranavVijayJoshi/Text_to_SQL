-- Runs automatically on first container start
-- Enables the pgvector extension needed for LangMem memory store

CREATE EXTENSION IF NOT EXISTS vector;

-- Confirm it worked
DO $$
BEGIN
    RAISE NOTICE 'pgvector extension enabled successfully';
END
$$;
