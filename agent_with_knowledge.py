"""HTTP API for querying the knowledge-backed Agno agent.

Run with:
    uvicorn agent_with_knowledge:app --host 0.0.0.0 --port 8000 --reload
"""

import os
from threading import Lock
from uuid import uuid4

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from agno.agent import Agent
from agno.knowledge.embedder.ollama import OllamaEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.models.ollama import Ollama
from agno.vectordb.pgvector import PgVector


db_url = os.getenv("DATABASE_URL", "postgresql+psycopg://ai:ai@localhost:5532/ai")
ollama_api_key = os.getenv("OLLAMA_API_KEY")

if not ollama_api_key:
    raise RuntimeError("Set the OLLAMA_API_KEY environment variable before starting the API.")

knowledge = Knowledge(
    vector_db=PgVector(
        table_name="Thai Recipes",
        db_url=db_url,
        embedder=OllamaEmbedder(id="nomic-embed-text", dimensions=768),
    ),
)

agent = Agent(
    model=Ollama(
        id="gpt-oss:120b-cloud",
        api_key=ollama_api_key,
    ),
    knowledge=knowledge,
    search_knowledge=True,
    markdown=True,
)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Text to search in the knowledge base")


class SearchResult(BaseModel):
    id: str
    query: str
    output: str


app = FastAPI(title="Knowledge Agent API")

# Results are intentionally kept in memory. They are cleared if the service restarts.
search_results: dict[str, SearchResult] = {}
results_lock = Lock()


def response_text(response: object) -> str:
    """Extract the readable content from Agno's response object."""
    content = getattr(response, "content", response)
    return content if isinstance(content, str) else str(content)


@app.post("/search", response_model=SearchResult, status_code=status.HTTP_201_CREATED)
def create_search(request: SearchRequest) -> SearchResult:
    """Search the knowledge base and store the generated answer."""
    try:
        response = agent.run(request.query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}") from exc

    result = SearchResult(
        id=str(uuid4()),
        query=request.query,
        output=response_text(response),
    )
    with results_lock:
        search_results[result.id] = result
    return result


@app.get("/search/{search_id}", response_model=SearchResult)
def get_search(search_id: str) -> SearchResult:
    """Return the saved output for a previous POST /search request."""
    with results_lock:
        result = search_results.get(search_id)

    if result is None:
        raise HTTPException(status_code=404, detail="Search result not found")
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
