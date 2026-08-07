from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.hackernews import HackerNewsTools
#report on trending startups and products
agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),  # or another Groq model
    tools=[HackerNewsTools()],
    markdown=True,
)

if __name__ == "__main__":
    agent.print_response(
        "Write a report on trending startups and products.", stream=True
    )