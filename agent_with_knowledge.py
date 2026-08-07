"""HTTP API for querying the knowledge-backed Agno agent.

Run with:
    uvicorn agent_with_knowledge:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import shutil
from pathlib import Path
from threading import Lock
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from agno.agent import Agent
from agno.knowledge.embedder.ollama import OllamaEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.knowledge.reader.pdf_reader import PDFReader
from agno.models.ollama import Ollama
from agno.vectordb.pgvector import PgVector


db_url = os.getenv("DATABASE_URL", "postgresql+psycopg://ai:ai@localhost:5532/ai")
ollama_api_key = os.getenv("OLLAMA_API_KEY")

if not ollama_api_key:
    raise RuntimeError("Set the OLLAMA_API_KEY environment variable before starting the API.")

knowledge = Knowledge(
    vector_db=PgVector(
        table_name="knowledgebase",
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
    # Retrieval is performed in create_search so it can be filtered by document ID.
    search_knowledge=False,
    markdown=True,
)


class SearchRequest(BaseModel):
    document_id: str = Field(..., min_length=1, description="ID returned by POST /documents")
    query: str = Field(..., min_length=1, description="Text to search in the knowledge base")


class SearchResult(BaseModel):
    id: str
    document_id: str
    query: str
    output: str


app = FastAPI(title="Knowledge Agent API")

# Results are intentionally kept in memory. They are cleared if the service restarts.
search_results: dict[str, SearchResult] = {}
results_lock = Lock()
knowledge_lock = Lock()
upload_directory = Path(__file__).parent / "uploaded_pdfs"
upload_directory.mkdir(exist_ok=True)


def response_text(response: object) -> str:
    """Extract the readable content from Agno's response object."""
    content = getattr(response, "content", response)
    return content if isinstance(content, str) else str(content)


class DocumentUploadResult(BaseModel):
    id: str
    filename: str
    message: str
    metadata: dict[str, str]


@app.post("/documents", response_model=DocumentUploadResult, status_code=status.HTTP_201_CREATED)
def upload_document(
    file: UploadFile = File(..., description="PDF document to add to the knowledge base"),
    department: str | None = Form(default=None),
    version: str | None = Form(default=None),
    user_tag: str | None = Form(default=None),
) -> DocumentUploadResult:
    """Save an uploaded PDF and add its pages to the knowledge base."""
    filename = Path(file.filename or "document.pdf").name
    is_pdf = filename.lower().endswith(".pdf") or file.content_type == "application/pdf"
    if not is_pdf:
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    document_id = str(uuid4())
    saved_path = upload_directory / f"{document_id}_{filename}"
    metadata = {
        key: value
        for key, value in {
            "department": department,
            "version": version,
            "user_tag": user_tag,
            "document_id": document_id
        }.items()
        if value is not None
    }

    try:
        with saved_path.open("wb") as destination:
            shutil.copyfileobj(file.file, destination)

        with knowledge_lock:
            knowledge.insert(
                name=filename,
                path=str(saved_path),
                reader=PDFReader(split_on_pages=True),
                metadata=metadata,
            )
    except Exception as exc:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"PDF upload failed: {exc}") from exc
    finally:
        file.file.close()

    result = DocumentUploadResult(
        id=document_id,
        filename=filename,
        message="PDF uploaded and added to the knowledge base",
        metadata=metadata,
    )
    with knowledge_lock:
        uploaded_documents[result.id] = result
    return result


@app.post("/search", response_model=SearchResult, status_code=status.HTTP_201_CREATED)
def create_search(request: SearchRequest) -> SearchResult:
    """Search the knowledge base and store the generated answer."""
    try:
        with knowledge_lock:
           
            documents = knowledge.search(
                request.query,
                filters={"document_id": request.document_id},
            )
            if not documents:
                raise HTTPException(
                    status_code=404,
                    detail="No relevant content was found in the selected PDF",
                )

            context = "\n\n".join(
                str(getattr(document, "content", document)) for document in documents
            )
            response = agent.run(
                "Answer the question using only the supplied PDF excerpts. "
                f"\n\nPDF excerpts:\n{context}\n\nQuestion: {request.query}"
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}") from exc

    result = SearchResult(
        id=str(uuid4()),
        document_id=request.document_id,
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
