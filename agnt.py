import os

from agno.agent import Agent
from agno.knowledge.embedder.ollama import OllamaEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.models.ollama import Ollama
from agno.vectordb.pgvector import PgVector

db_url = os.getenv("DATABASE_URL", "postgresql+psycopg://ai:ai@localhost:5532/ai")
ollama_api_key = os.getenv("OLLAMA_API_KEY")

if not ollama_api_key:
    raise RuntimeError("Set the OLLAMA_API_KEY environment variable before running this script.")

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
    search_knowledge=True,
    markdown=True,
)

if __name__ == "__main__":
    id = 54765876
    knowledge.insert(
        name="Thai Recipes",
        url="https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf",
        metadata={"id": id}
    )
    knowledge.insert(
        name="Agno Cookbook",
        url="https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf",  # replace with your second PDF URL
    )
    agent.print_response(
        "Search the knowledge base and summarize both documents."
    )
