"""Supervisor agent for coordinating sub-agents."""
from typing import List, Callable
from langchain_core.language_models import BaseChatModel
from models import AgentInfo
from .base_agent import BaseAgent


class SupervisorAgent(BaseAgent):
    """Supervisor agent that routes tasks to specialized sub-agents."""

    def __init__(self, llm: BaseChatModel, available_agents: list[AgentInfo]):
        """Initialize the supervisor agent."""
        self.available_agents = available_agents
        super().__init__(llm)

    @property
    def tools(self) -> List[Callable]:
        """Return a list of tools for the agent."""
        return []

    @property
    def system_prompt(self) -> str:
        """Return the system prompt for the agent."""
        agent_descriptions = "\n".join(
            f"- {agent.name}: {agent.description}" 
            for agent in self.available_agents
        )
        
        return f"""You are a supervisor coordinating a team of specialized agents.

Available agents:
{agent_descriptions}

Your job is to analyze the current task and conversation history, then decide which agent 
should handle the next step. Choose FINISH when the task is complete.

Consider:
- What has already been done by previous agents
- What remains to be done
- Which agent is best suited for the next step"""
