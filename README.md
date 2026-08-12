# Tenant RAG API

This FastAPI service ingests PDFs into PostgreSQL/pgvector, retrieves tenant-scoped chunks, and uses Ollama to answer questions with page citations.

## Setup

Start PostgreSQL with pgvector, set environment variables, then run the API from this folder:

```powershell
$env:DATABASE_URL = "postgresql+psycopg://rag:rag_password@localhost:5432/rag"
$env:OLLAMA_API_KEY = "your-ollama-api-key"
$env:RAG_API_KEYS = "dev-api-key:company-1"
python -m uvicorn agent_with_knowledge:app --reload
```

The API expects an `X-API-Key` header. For the example above, use `dev-api-key`; its tenant is `company-1`.

Open interactive documentation at `http://127.0.0.1:8000/docs`.

## API flow

1. `POST /documents` uploads a PDF and returns `202 queued`.
2. The background task validates, extracts, chunks, embeds, and indexes page text. Its lifecycle is `queued → processing → ready` or `failed`.
3. `GET /documents/{document_id}` reports status.
4. `POST /search` retrieves only chunks owned by the API key's company and sends them to Ollama.
5. The response includes an answer, source excerpts/page numbers, and a `cached` flag. Identical questions for the same document are cached for one hour.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/documents` | Upload and queue a PDF. Form fields: `file`, optional `page_start`, `page_end`, `replace_existing`. |
| `GET` | `/documents` | List documents for the authenticated company. |
| `GET` | `/documents/{document_id}` | Read one document's ingestion status. |
| `DELETE` | `/documents/{document_id}` | Delete the company's vectors, local file, metadata, and cache entries. |
| `POST` | `/search` | Ask a question. JSON: `query`, optional `document_id`, optional `top_k`. |
| `POST` | `/documents/{document_id}/extract` | Generate a schema-validated maintenance checklist. |
| `GET` | `/health` | Liveness check. |
| `GET` | `/health/database` | PostgreSQL connectivity check. |

Every request except `/health` and `/health/database` requires:

```http
X-API-Key: dev-api-key
```

## Search example

```json
{
  "document_id": "uploaded-document-uuid",
  "query": "What are the required maintenance intervals?",
  "top_k": 5
}
```

Only a company ID derived from the API key is used for retrieval; clients cannot select another tenant. Every vector stores `company_id`, `document_id`, `file_name`, `page_number`, and `chunk_index`.
