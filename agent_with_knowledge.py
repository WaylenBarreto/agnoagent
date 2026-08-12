"""Tenant-isolated PDF RAG API backed by PostgreSQL/pgvector and Ollama."""

import json
import logging
import os
import shutil
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from threading import Lock
from time import perf_counter, time
from typing import Literal
from uuid import uuid4

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, UploadFile, status
from pydantic import BaseModel, ConfigDict, Field
from pypdf import PdfReader
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from agno.agent import Agent
from agno.knowledge.document.base import Document
from agno.knowledge.embedder.ollama import OllamaEmbedder
from agno.models.ollama import Ollama
from agno.vectordb.pgvector import PgVector


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_api")

MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024
MAX_PAGE_COUNT = 300
CHUNK_SIZE = 1_200
CHUNK_OVERLAP = 200
CACHE_TTL_SECONDS = 60 * 60
MIN_SIMILARITY_SCORE = 0.35
UPLOAD_DIRECTORY = Path(__file__).parent / "uploaded_pdfs"
UPLOAD_DIRECTORY.mkdir(exist_ok=True)


class SourceCitation(BaseModel):
    document_id: str
    file_name: str
    page_number: int
    chunk_index: int
    excerpt: str
    score: float


class DocumentUploadResponse(BaseModel):
    document_id: str
    company_id: str
    file_name: str
    status: Literal["queued", "processing", "ready", "failed"]
    page_count: int


class DocumentStatusResponse(DocumentUploadResponse):
    chunk_count: int = 0
    created_at: datetime
    updated_at: datetime
    error_message: str | None = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4_000)
    document_id: str | None = Field(default=None, description="Limit retrieval to one document")
    top_k: int = Field(default=5, ge=1, le=10)


class SearchResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
    cached: bool


class ExtractionRequest(BaseModel):
    type: Literal["checklist"]


class ChecklistItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    description: str
    frequency: str | None = None
    source_page: int


class ExtractionResponse(BaseModel):
    document_id: str
    type: Literal["checklist"]
    items: list[ChecklistItem]
    cached: bool = False


class DocumentRepository:
    """Parameterized PostgreSQL access for document lifecycle metadata."""

    def __init__(self, database_url: str) -> None:
        self.engine: Engine = create_engine(database_url, pool_pre_ping=True)
        self.ensure_schema()

    def ensure_schema(self) -> None:
        statement = text("""
            CREATE TABLE IF NOT EXISTS documents (
                document_id VARCHAR(36) PRIMARY KEY,
                company_id VARCHAR(128) NOT NULL,
                file_name TEXT NOT NULL,
                stored_path TEXT NOT NULL,
                status VARCHAR(16) NOT NULL,
                page_count INTEGER NOT NULL DEFAULT 0,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMPTZ NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL,
                error_message TEXT
            );
            CREATE INDEX IF NOT EXISTS documents_company_id_idx ON documents (company_id);
        """)
        with self.engine.begin() as connection:
            for sql in statement.text.split(";"):
                if sql.strip():
                    connection.execute(text(sql))

    def create(self, *, document_id: str, company_id: str, file_name: str, stored_path: str, page_count: int) -> None:
        now = datetime.now(timezone.utc)
        with self.engine.begin() as connection:
            connection.execute(text("""
                INSERT INTO documents (document_id, company_id, file_name, stored_path, status, page_count, created_at, updated_at)
                VALUES (:document_id, :company_id, :file_name, :stored_path, 'queued', :page_count, :created_at, :updated_at)
            """), {"document_id": document_id, "company_id": company_id, "file_name": file_name,
                  "stored_path": stored_path, "page_count": page_count, "created_at": now, "updated_at": now})

    def update_status(self, document_id: str, company_id: str, status_value: str, *, chunk_count: int = 0, error_message: str | None = None) -> None:
        with self.engine.begin() as connection:
            connection.execute(text("""
                UPDATE documents
                SET status = :status, chunk_count = :chunk_count, error_message = :error_message, updated_at = :updated_at
                WHERE document_id = :document_id AND company_id = :company_id
            """), {"status": status_value, "chunk_count": chunk_count, "error_message": error_message,
                  "updated_at": datetime.now(timezone.utc), "document_id": document_id, "company_id": company_id})

    def get(self, document_id: str, company_id: str) -> dict | None:
        with self.engine.connect() as connection:
            row = connection.execute(text("""
                SELECT document_id, company_id, file_name, stored_path, status, page_count, chunk_count,
                       created_at, updated_at, error_message
                FROM documents WHERE document_id = :document_id AND company_id = :company_id
            """), {"document_id": document_id, "company_id": company_id}).mappings().first()
        return dict(row) if row else None

    def list(self, company_id: str) -> list[dict]:
        with self.engine.connect() as connection:
            rows = connection.execute(text("""
                SELECT document_id, company_id, file_name, stored_path, status, page_count, chunk_count,
                       created_at, updated_at, error_message
                FROM documents WHERE company_id = :company_id ORDER BY created_at DESC
            """), {"company_id": company_id}).mappings().all()
        return [dict(row) for row in rows]

    def delete(self, document_id: str, company_id: str) -> dict | None:
        document = self.get(document_id, company_id)
        if document is None:
            return None
        with self.engine.begin() as connection:
            connection.execute(text("DELETE FROM documents WHERE document_id = :document_id AND company_id = :company_id"),
                               {"document_id": document_id, "company_id": company_id})
        return document

    def health(self) -> bool:
        with self.engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True


class KnowledgeAgent:
    """Owns embeddings, pgvector retrieval, and Ollama answer generation."""

    model_name = "gpt-oss:120b-cloud"

    def __init__(self) -> None:
        self.database_url = os.getenv("DATABASE_URL", "postgresql+psycopg://ai:ai@localhost:5532/ai")
        api_key = os.getenv("OLLAMA_API_KEY")
        if not api_key:
            raise RuntimeError("Set OLLAMA_API_KEY before using the RAG API.")
        self.vector_db = PgVector(
            table_name="knowledgebase",
            db_url=self.database_url,
            embedder=OllamaEmbedder(id="nomic-embed-text", dimensions=768),
        )
        self.agent = Agent(model=Ollama(id=self.model_name, api_key=api_key), markdown=True)

    @staticmethod
    def _chunks(text_value: str) -> list[str]:
        clean_text = " ".join(text_value.split())
        return [clean_text[start:start + CHUNK_SIZE] for start in range(0, len(clean_text), CHUNK_SIZE - CHUNK_OVERLAP)]

    def add_document(self, *, document_id: str, company_id: str, file_name: str, file_path: Path,
                     page_start: int | None, page_end: int | None, replace_existing: bool) -> tuple[int, int]:
        """Extract selected PDF pages, chunk them, embed them, and store tenant metadata."""
        reader = PdfReader(str(file_path))
        page_count = len(reader.pages)
        if page_count == 0:
            raise ValueError("The PDF has no pages")
        start = page_start or 1
        end = page_end or page_count
        if start < 1 or end < start or end > page_count:
            raise ValueError("page_range must be within the PDF page count")
        if replace_existing:
            self.vector_db.delete_by_metadata({"company_id": company_id, "document_id": document_id})

        chunks: list[Document] = []
        for page_number in range(start, end + 1):
            page_text = reader.pages[page_number - 1].extract_text() or ""
            for chunk_index, chunk in enumerate(self._chunks(page_text)):
                chunks.append(Document(
                    id=f"{document_id}:{page_number}:{chunk_index}", name=file_name, content=chunk,
                    meta_data={"company_id": company_id, "document_id": document_id, "file_name": file_name,
                               "page_number": page_number, "chunk_index": chunk_index},
                ))
        if not chunks:
            raise ValueError("The selected PDF pages contain no extractable text")
        self.vector_db.insert(content_hash=document_id, documents=chunks)
        return page_count, len(chunks)

    def search(self, *, company_id: str, document_id: str | None, query: str, top_k: int) -> tuple[str, list[SourceCitation]]:
        filters: dict[str, str] = {"company_id": company_id}
        if document_id:
            filters["document_id"] = document_id
        documents = self.vector_db.search(query, limit=top_k, filters=filters)
        sources = [SourceCitation(
            document_id=str(document.meta_data["document_id"]), file_name=str(document.meta_data["file_name"]),
            page_number=int(document.meta_data["page_number"]), chunk_index=int(document.meta_data["chunk_index"]),
            excerpt=document.content[:500], score=float(document.meta_data.get("similarity_score", 0)),
        ) for document in documents]
        if not sources or sources[0].score < MIN_SIMILARITY_SCORE:
            raise LookupError("The answer was not found in the selected document context")
        context = "\n\n".join(f"[Source {index + 1}, page {source.page_number}] {source.excerpt}" for index, source in enumerate(sources))
        response = self.agent.run(f"Answer only from these PDF excerpts. If insufficient, say so.\n\n{context}\n\nQuestion: {query}")
        content = getattr(response, "content", response)
        return (content if isinstance(content, str) else str(content)), sources

    def extract_checklist(self, *, company_id: str, document_id: str) -> list[ChecklistItem]:
        _, sources = self.search(company_id=company_id, document_id=document_id,
                                 query="maintenance checklist tasks, schedules, and procedures", top_k=10)
        context = "\n".join(f"Page {source.page_number}: {source.excerpt}" for source in sources)
        response = self.agent.run("Create a maintenance checklist from this context. Return only JSON: "
                                  "[{\"title\": str, \"description\": str, \"frequency\": str|null, \"source_page\": int}].\n" + context)
        raw = getattr(response, "content", response)
        payload = str(raw).strip().removeprefix("```json").removesuffix("```").strip()
        return [ChecklistItem.model_validate(item) for item in json.loads(payload)]


@lru_cache
def get_repository() -> DocumentRepository:
    return DocumentRepository(os.getenv("DATABASE_URL", "postgresql+psycopg://ai:ai@localhost:5532/ai"))


@lru_cache
def get_knowledge_agent() -> KnowledgeAgent:
    return KnowledgeAgent()


def current_company_id(x_api_key: str = Header(..., alias="X-API-Key")) -> str:
    """Map API keys to tenants; replace this with JWT/session auth in production."""
    configured_keys = os.getenv("RAG_API_KEYS", "")
    key_map = dict(item.split(":", 1) for item in configured_keys.split(",") if ":" in item)
    company_id = key_map.get(x_api_key)
    if not company_id:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return company_id


answer_cache: dict[tuple[str, str, str, str], tuple[float, SearchResponse]] = {}
cache_lock = Lock()


def normalized_query(query: str) -> str:
    return " ".join(query.casefold().split())


def cache_key(company_id: str, document_id: str | None, query: str) -> tuple[str, str, str, str]:
    return company_id, document_id or "all", normalized_query(query), KnowledgeAgent.model_name


def cache_get(key: tuple[str, str, str, str]) -> SearchResponse | None:
    with cache_lock:
        cached = answer_cache.get(key)
        if cached and time() - cached[0] < CACHE_TTL_SECONDS:
            return cached[1].model_copy(update={"cached": True})
        answer_cache.pop(key, None)
    return None


def cache_put(key: tuple[str, str, str, str], response: SearchResponse) -> None:
    with cache_lock:
        answer_cache[key] = (time(), response)


def invalidate_document_cache(company_id: str, document_id: str) -> None:
    with cache_lock:
        for key in list(answer_cache):
            if key[0] == company_id and (key[1] == document_id or key[1] == "all"):
                del answer_cache[key]


def document_response(document: dict) -> DocumentStatusResponse:
    return DocumentStatusResponse(**{key: value for key, value in document.items() if key != "stored_path"})


def ingest_document(document_id: str, company_id: str, page_start: int | None, page_end: int | None,
                    replace_existing: bool) -> None:
    repository = get_repository()
    document = repository.get(document_id, company_id)
    if document is None:
        return
    started = perf_counter()
    repository.update_status(document_id, company_id, "processing")
    try:
        _, chunk_count = get_knowledge_agent().add_document(
            document_id=document_id, company_id=company_id, file_name=document["file_name"],
            file_path=Path(document["stored_path"]), page_start=page_start, page_end=page_end,
            replace_existing=replace_existing,
        )
        repository.update_status(document_id, company_id, "ready", chunk_count=chunk_count)
        invalidate_document_cache(company_id, document_id)
        logger.info("ingestion complete document_id=%s company_id=%s chunks=%s seconds=%.2f", document_id, company_id, chunk_count, perf_counter() - started)
    except Exception as exc:
        logger.exception("ingestion failed document_id=%s company_id=%s", document_id, company_id)
        repository.update_status(document_id, company_id, "failed", error_message=str(exc)[:2_000])


app = FastAPI(title="Tenant RAG API")


@app.post("/documents", response_model=DocumentUploadResponse, status_code=status.HTTP_202_ACCEPTED)
def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    page_start: int | None = Form(default=None, ge=1),
    page_end: int | None = Form(default=None, ge=1),
    replace_existing: bool = Form(default=False),
    company_id: str = Depends(current_company_id),
    repository: DocumentRepository = Depends(get_repository),
) -> DocumentUploadResponse:
    filename = Path(file.filename or "document.pdf").name
    if not filename.lower().endswith(".pdf") or file.content_type not in {"application/pdf", "application/x-pdf", None}:
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")
    document_id = str(uuid4())
    stored_path = UPLOAD_DIRECTORY / f"{document_id}_{filename}"
    size = 0
    try:
        with stored_path.open("wb") as destination:
            while block := file.file.read(1024 * 1024):
                size += len(block)
                if size > MAX_FILE_SIZE_BYTES:
                    raise HTTPException(status_code=413, detail="PDF exceeds the 20 MB limit")
                destination.write(block)
        reader = PdfReader(str(stored_path))
        page_count = len(reader.pages)
        if page_count == 0 or page_count > MAX_PAGE_COUNT:
            raise HTTPException(status_code=400, detail=f"PDF must contain between 1 and {MAX_PAGE_COUNT} pages")
        if page_start and page_end and page_end < page_start:
            raise HTTPException(status_code=400, detail="page_end must be greater than or equal to page_start")
        if (page_start and page_start > page_count) or (page_end and page_end > page_count):
            raise HTTPException(status_code=400, detail="page_range exceeds the PDF page count")
        repository.create(document_id=document_id, company_id=company_id, file_name=filename,
                          stored_path=str(stored_path), page_count=page_count)
    except HTTPException:
        stored_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Invalid PDF: {exc}") from exc
    finally:
        file.file.close()
    background_tasks.add_task(ingest_document, document_id, company_id, page_start, page_end, replace_existing)
    return DocumentUploadResponse(document_id=document_id, company_id=company_id, file_name=filename, status="queued", page_count=page_count)


@app.get("/documents", response_model=list[DocumentStatusResponse])
def list_documents(company_id: str = Depends(current_company_id), repository: DocumentRepository = Depends(get_repository)) -> list[DocumentStatusResponse]:
    return [document_response(document) for document in repository.list(company_id)]


@app.get("/documents/{document_id}", response_model=DocumentStatusResponse)
def get_document(document_id: str, company_id: str = Depends(current_company_id), repository: DocumentRepository = Depends(get_repository)) -> DocumentStatusResponse:
    document = repository.get(document_id, company_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return document_response(document)


@app.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: str, company_id: str = Depends(current_company_id), repository: DocumentRepository = Depends(get_repository), knowledge_agent: KnowledgeAgent = Depends(get_knowledge_agent)) -> None:
    document = repository.delete(document_id, company_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    if not knowledge_agent.vector_db.delete_by_metadata({"company_id": company_id, "document_id": document_id}):
        logger.warning("vector deletion failed document_id=%s", document_id)
    Path(document["stored_path"]).unlink(missing_ok=True)
    invalidate_document_cache(company_id, document_id)


@app.post("/search", response_model=SearchResponse)
def search_documents(request: SearchRequest, company_id: str = Depends(current_company_id), repository: DocumentRepository = Depends(get_repository), knowledge_agent: KnowledgeAgent = Depends(get_knowledge_agent)) -> SearchResponse:
    if request.document_id:
        document = repository.get(request.document_id, company_id)
        if document is None:
            raise HTTPException(status_code=404, detail="Document not found")
        if document["status"] != "ready":
            raise HTTPException(status_code=409, detail=f"Document is {document['status']}")
    key = cache_key(company_id, request.document_id, request.query)
    if cached := cache_get(key):
        logger.info("cache hit company_id=%s document_id=%s", company_id, request.document_id)
        return cached
    started = perf_counter()
    try:
        answer, sources = knowledge_agent.search(company_id=company_id, document_id=request.document_id, query=request.query, top_k=request.top_k)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("search failed company_id=%s document_id=%s", company_id, request.document_id)
        raise HTTPException(status_code=500, detail="Search failed") from exc
    response = SearchResponse(answer=answer, sources=sources, cached=False)
    cache_put(key, response)
    logger.info("cache miss company_id=%s document_id=%s sources=%s seconds=%.2f", company_id, request.document_id, len(sources), perf_counter() - started)
    return response


@app.post("/documents/{document_id}/extract", response_model=ExtractionResponse)
def extract_document(document_id: str, request: ExtractionRequest, company_id: str = Depends(current_company_id), repository: DocumentRepository = Depends(get_repository), knowledge_agent: KnowledgeAgent = Depends(get_knowledge_agent)) -> ExtractionResponse:
    document = repository.get(document_id, company_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    if document["status"] != "ready":
        raise HTTPException(status_code=409, detail=f"Document is {document['status']}")
    try:
        return ExtractionResponse(document_id=document_id, type=request.type,
                                  items=knowledge_agent.extract_checklist(company_id=company_id, document_id=document_id))
    except Exception as exc:
        logger.exception("extraction failed document_id=%s", document_id)
        raise HTTPException(status_code=500, detail="Checklist extraction failed") from exc


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/database")
def database_health(repository: DocumentRepository = Depends(get_repository)) -> dict[str, str]:
    try:
        repository.health()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Database unavailable") from exc
    return {"status": "ok", "database": "reachable"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
