import os
from pathlib import Path
#pdf summarizer

from agno.agent import Agent
from agno.knowledge.reader.pdf_reader import PDFReader
from agno.models.groq import Groq

pdf_path = Path(r"C:\Users\Admin\Desktop\climate.pdf")

agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    markdown=True,
)

if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("Set the GROQ_API_KEY environment variable before running this script.")

    documents = PDFReader(split_on_pages=True).read(pdf_path, name=pdf_path.name)
    text = "\n\n".join(doc.content for doc in documents if getattr(doc, "content", None))

    agent.print_response(f"Summarize this document:\n\n{text[:12000]}")