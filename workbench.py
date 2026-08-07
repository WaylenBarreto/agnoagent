from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.os import AgentOS
from agno.tools.workspace import Workspace

workbench = Agent(
    name="Workbench",
    model="cerebras:gemma-4-31b",
    db=SqliteDb(db_file="workbench.db"),
    tools=[Workspace(".")],
    enable_agentic_memory=True,
    add_history_to_context=True,
    num_history_runs=3,
)

agent_os = AgentOS(agents=[workbench], tracing=True)
app = agent_os.get_app()

if __name__ == "__main__":
    agent_os.serve(app="workbench:app", reload=True)