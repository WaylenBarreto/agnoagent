from pathlib import Path
from agno.agent import Agent
from agno.tools.workspace import Workspace

folder = Path(__file__).parent

sorting_hat = Agent(
    name="Sorting Hat",
    model="cerebras:gemma-4-31b",
    tools=[Workspace(root=str(folder), allowed=["read", "list", "search"])],
    instructions=(
        "Walk the folder, figure out what's there, and propose a clean organization. "
        "Decide the categories yourself. Return a tidy summary, a category breakdown, "
        "and a folder tree."
    ),
    markdown=True,
)

sorting_hat.print_response(f"Inventory and organize {folder}", stream=True)