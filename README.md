# Agno agent examples

This folder contains small examples built with the [Agno](https://www.agno.com/) framework. The scripts use different models, tools, and knowledge sources.

## Setup

Create and activate a virtual environment, then install the packages needed by the script you want to run. The knowledge API also needs FastAPI and Uvicorn:

```powershell
python -m pip install agno fastapi "uvicorn[standard]"
```

Credentials are read from environment variables and must not be committed. `.env.example` lists the required variable names; set their values in the shell before running a script. For PowerShell:

```powershell
$env:OLLAMA_API_KEY = "your-ollama-api-key"
# Optional: override the default PostgreSQL connection
$env:DATABASE_URL = "postgresql+psycopg://user:password@localhost:5532/database"
```

`.env` and common key files are ignored by Git. Existing keys that were previously committed should be revoked and replaced; adding a file to `.gitignore` does not remove secrets from Git history.

## Python files

| File | What it does | How to run |
| --- | --- | --- |
| `agent_with_knowledge.py` | Runs a FastAPI service that accepts PDF uploads, indexes them in a PostgreSQL/pgvector knowledge base, and answers questions with an Ollama model. `POST /documents` uploads and indexes a PDF; `POST /search` asks a question about a selected document ID; `GET /search/{id}` returns a saved result. Results are kept only in memory while the server runs. | `python -m uvicorn agent_with_knowledge:app --reload` |
| `agnt.py` | Inserts two PDF sources into the `thai_recipes` pgvector knowledge base, then asks the Ollama agent for a combined summary. | `python agnt.py` |
| `news.py` | Uses Groq and the Hacker News tool to generate a streamed report about trending startups and products. Requires `GROQ_API_KEY`. | `python news.py` |
| `pdf.py` | Reads the local `C:\Users\Admin\Desktop\climate.pdf` one page at a time and asks a Groq model to summarize its text. Requires `GROQ_API_KEY` and that PDF file. | `python pdf.py` |
| `sortinghat.py` | Gives an agent read-only workspace tools to inspect this folder and propose a clearer organization, including a tree and category breakdown. | `python sortinghat.py` |
| `workbench.py` | Creates an AgentOS application named Workbench with SQLite-backed agent memory and workspace access. It serves the AgentOS web/API app. | `python workbench.py` |

## Using the knowledge API

Start the API from this folder:

```powershell
python -m uvicorn agent_with_knowledge:app --reload
```

### Upload a PDF

In Postman, create this request:

- Method: `POST`
- URL: `http://127.0.0.1:8000/documents`
- Body: select **form-data**
- Add a field named `file`, change its type from **Text** to **File**, then choose your PDF.
- Optional text fields: `department`, `version`, and `user_tag`.

For example, the equivalent metadata fields are:

| Key | Value |
| --- | --- |
| `file` | your PDF file |
| `department` | `engineering` |
| `version` | `2.1` |
| `user_tag` | `custom` |

The API returns a document ID after it has saved and indexed the PDF. Uploaded PDFs are stored locally in `uploaded_pdfs/`, which is ignored by Git.

If no `file` form-data field is sent, FastAPI returns `422 Unprocessable Content`. If a non-PDF is sent, the API returns `400 Only PDF files are accepted`.

### Ask a question

After the upload succeeds, copy the PDF `id` returned by `POST /documents`. Include it with every `POST http://127.0.0.1:8000/search` request; this ensures the search is limited to that uploaded PDF:

```json
{
  "document_id": "paste-the-id-returned-by-documents-here",
  "query": "What are the main conclusions of this PDF?"
}
```

The response contains a search `id`, `document_id`, the submitted `query`, and the generated `output`. Retrieve the saved result with:

```http
GET http://127.0.0.1:8000/search/<id>
```

Interactive endpoint documentation is available at `http://127.0.0.1:8000/docs`.

If you try to search before a PDF has been uploaded since the server started, the API returns this clear error:

```json
{
  "detail": "No PDF has been uploaded. Upload a PDF with POST /documents before searching."
}
```

## How `agent_with_knowledge.py` works

- `Knowledge(...)` configures the pgvector database where PDF text is embedded and stored. `OllamaEmbedder` converts each text chunk into an embedding for semantic search.
- `Agent(...)` configures the Ollama language model. The API retrieves the relevant chunks itself so it can restrict them to the requested document ID, then passes those excerpts to the agent.
- `upload_document(...)` is the `POST /documents` function. It receives the `file` from Postman, checks that it is a PDF, saves it in `uploaded_pdfs/`, and calls `knowledge.insert(...)`. `PDFReader(split_on_pages=True)` reads the PDF a page at a time. The optional metadata fields are stored with the indexed chunks.
- `create_search(...)` is the `POST /search` function. It first checks the supplied `document_id`, retrieves matching chunks only from that PDF, and provides them to `agent.run(...)` with your `query`. The returned answer is saved under a generated search ID and sent back in `output`.
- `get_search(...)` is the `GET /search/{search_id}` function. It looks up and returns a result produced earlier by `create_search(...)`; unknown IDs return `404`.
- `response_text(...)` extracts the readable answer from Agno's response object.
- `knowledge_lock` prevents an upload from modifying the knowledge base while another request is searching it. `results_lock` protects the in-memory saved search results.
