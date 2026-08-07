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
| `agent_with_knowledge.py` | Runs a FastAPI service that asks an Ollama model questions using a PostgreSQL/pgvector knowledge base. `POST /search` accepts a query and returns the answer; `GET /search/{id}` returns a saved result. Results are kept only in memory while the server runs. | `python -m uvicorn agent_with_knowledge:app --reload` |
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

In Postman, submit a question:

```http
POST http://127.0.0.1:8000/search
Content-Type: application/json

{
  "query": "What Thai recipes are available in the knowledge base?"
}
```

The response contains an `id`, the submitted `query`, and the generated `output`. Retrieve the saved result with:

```http
GET http://127.0.0.1:8000/search/<id>
```

Interactive endpoint documentation is available at `http://127.0.0.1:8000/docs`.
