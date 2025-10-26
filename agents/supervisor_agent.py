"""Supervisor agent for coordinating sub-agents."""
from typing import List, Callable
from langchain_core.language_models import BaseChatModel
from .base_agent import BaseAgent


class SupervisorAgent(BaseAgent):
    """Supervisor agent that routes tasks to specialized sub-agents."""

    def __init__(self, llm: BaseChatModel, available_agents: list[BaseAgent]):
        """Initialize the supervisor agent."""
        self.available_agents = available_agents
        super().__init__(llm)

    @property
    def description(self) -> str:
        """Return a description of the agent."""
        return "Acts as a supervisor to coordinate a team of specialized agents."

    @property
    def tools(self) -> List[Callable]:
        """Return a list of tools for the agent."""
        return []

    @property
    def system_prompt(self) -> str:
        """Return the system prompt for the agent."""
        agent_descriptions = "\n".join(
            f"- {agent.__class__.__name__}: {agent.description}" 
            for agent in self.available_agents
        )
        
        return f"""You are a supervisor orchestrating a team of specialized agents. Your role is to analyze the user's task and the ongoing conversation to delegate the next step to the most appropriate agent.

Here are the agents available to you:
{agent_descriptions}

Follow these rules:
1.  **Analyze the Request**: Carefully read the user's task and the conversation history.
2.  **Break Down the Task**: If the task is complex, break it down into smaller, sequential steps. Each step should be handled by the most appropriate agent.
3.  **Delegate**: Choose the best agent to perform the next action. The agent's description will help you decide. For example, if the user asks to research a topic, you should route to the "ResearchAgent".
4.  **FINISH**: Once all steps of the task are fully completed and the user's request has been met, you must choose "FINISH". Do not finish if there are still steps to be done.
5.  **No Assumptions**: Do not make assumptions about what has been done. Base your decisions only on the conversation history. If the history is empty, start from the beginning of the task.

Your job is to decide which agent should act next. Choose FINISH ONLY when the entire task is complete."""
