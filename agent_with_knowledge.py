from agno.agent import Agent
from agno.knowledge.embedder.ollama import OllamaEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.models.ollama import Ollama
from agno.vectordb.pgvector import PgVector

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

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
        api_key="bded4aa1f6d843afa989af00268b7d2b._kJpuuusaxvrCpJiOqMt7sax",
    ),
    knowledge=knowledge,
    search_knowledge=True,
    markdown=True,
)

if __name__ == "__main__":
    knowledge.insert(
        url="https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"
    )
    agent.print_response(
        "Search the knowledge base and summarize the document."
    )