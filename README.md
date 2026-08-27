# Tenant-Isolated PDF RAG API

A portfolio project demonstrating a multi-tenant Retrieval-Augmented Generation API for asking questions about PDF documents.

The service combines FastAPI, PostgreSQL/pgvector, Agno, Ollama embeddings, and an Ollama language model. It supports document upload, asynchronous indexing, tenant-scoped retrieval, page-level citations, response caching, and structured maintenance-checklist extraction.

## What It Demonstrates

- Tenant isolation at the API and vector metadata-filtering layers
- PDF validation with file-size, page-count, and page-range limits
- Document lifecycle tracking: `queued`, `processing`, `ready`, and `failed`
- Page-aware chunk metadata for explainable source citations
- Semantic search with a configurable result limit
- Schema-validated checklist extraction using Pydantic
- PostgreSQL parameterized queries through SQLAlchemy
- Cache invalidation when a document is deleted or re-indexed

## Architecture

```mermaid
flowchart LR
    Client[API client] --> API[FastAPI]
    API --> Auth[API-key tenant mapping]
    API --> Metadata[(PostgreSQL metadata)]
    API --> Files[Local PDF storage]
    API --> Queue[FastAPI background task]
    Queue --> PDF[PDF extraction and chunking]
    PDF --> Embed[Ollama embeddings]
    Embed --> Vector[(pgvector)]
    API --> Vector
    API --> Model[Ollama language model]
    Vector --> Model
    Model --> API
```

## Request Flow

1. `POST /documents` validates and stores a PDF, creates document metadata, and returns `202 Accepted`.
2. A background task extracts page text, chunks it, creates embeddings, and stores tenant metadata in pgvector.
3. The document status changes from `queued` to `processing`, then `ready` or `failed`.
4. `POST /search` retrieves only chunks belonging to the authenticated tenant and sends them to the model with source context.
5. The response includes the answer, source excerpts, page numbers, similarity scores, and cache status.

## Requirements

- Python 3.11 or newer
- PostgreSQL with the pgvector extension
- An Ollama API key
- The `nomic-embed-text` embedding model configured for the Ollama account

Install the project dependencies in your virtual environment using the dependency file supplied with your project, then run the API from this directory.

## Local Setup

Start PostgreSQL with pgvector and configure the environment:

```powershell
$env:DATABASE_URL = "postgresql+psycopg://rag:rag_password@localhost:5432/rag"
$env:OLLAMA_API_KEY = "your-ollama-api-key"
$env:RAG_API_KEYS = "dev-api-key:company-1"
python -m uvicorn agent_with_knowledge:app --reload
```

The example API key maps requests to tenant `company-1`. Open the interactive API documentation at <http://127.0.0.1:8000/docs>.

For a real deployment, use a secrets manager and replace the demo API-key mapping with an identity provider. Do not commit `.env` files, credentials, tokens, uploaded PDFs, or database files.

## API Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/documents` | Upload and queue a PDF. Form fields: `file`, optional `page_start`, `page_end`, and `replace_existing`. |
| `GET` | `/documents` | List documents for the current tenant. |
| `GET` | `/documents/{document_id}` | Read document ingestion status. |
| `DELETE` | `/documents/{document_id}` | Delete tenant metadata, vectors, local file, and cache entries. |
| `POST` | `/search` | Ask a question using tenant-scoped retrieval. |
| `POST` | `/documents/{document_id}/extract` | Generate a schema-validated maintenance checklist. |
| `GET` | `/health` | Process liveness check. |
| `GET` | `/health/database` | PostgreSQL connectivity check. |

All document and search endpoints require an `X-API-Key` header:

```http
X-API-Key: dev-api-key
```

## Search Example

```json
{
  "document_id": "uploaded-document-uuid",
  "query": "What are the required maintenance intervals?",
  "top_k": 5
}
```

Example response shape:

```json
{
  "answer": "The required maintenance interval is ...",
  "sources": [
    {
      "document_id": "uploaded-document-uuid",
      "file_name": "manual.pdf",
      "page_number": 12,
      "chunk_index": 0,
      "excerpt": "...",
      "score": 0.81
    }
  ],
  "cached": false
}
```

## Project Structure

| File | Responsibility |
| --- | --- |
| `agent_with_knowledge.py` | FastAPI application, document lifecycle, retrieval, and extraction |
| `.env.example` | Local environment variable template |
| `uploaded_pdfs/` | Local development upload directory; ignored by Git |

## Current Scope and Limitations

This is a production-minded prototype intended for demonstration and portfolio use. The current implementation uses:

- FastAPI `BackgroundTasks` instead of a durable job queue
- Local filesystem storage instead of object storage
- Static environment-configured API keys instead of OAuth2 or JWT
- An in-memory, per-process response cache
- Runtime schema creation instead of migration tooling

These choices keep the project easy to run locally, but they are not sufficient for high availability, horizontal scaling, or sensitive production workloads.

## Production Roadmap

1. Move ingestion to a durable worker queue with retries and dead-letter handling.
2. Store PDFs in encrypted object storage with malware scanning and retention policies.
3. Replace static API keys with an identity provider and enforce database-level tenant isolation.
4. Add Redis caching, rate limiting, quotas, structured logging, metrics, and tracing.
5. Add Alembic migrations, automated tests, security tests, load tests, and retrieval-quality evaluations.
6. Add readiness checks for PostgreSQL, pgvector, the queue, embeddings, and the language model.

## Security Notes

Retrieved PDF content is untrusted input. The application should be deployed with prompt-injection testing, model timeouts, request limits, secret management, and redacted logs before handling sensitive documents.
