"""Research agent for information gathering."""
from typing import List, Callable
from .base_agent import BaseAgent
from tools import research_tool


class ResearchAgent(BaseAgent):
    """Agent specializing in information gathering."""

    @property
    def tools(self) -> List[Callable]:
        """Return a list of tools for the agent."""
        return [research_tool]

    @property
    def system_prompt(self) -> str:
        """Return the system prompt for the agent."""
        return """You are a research agent specializing in information gathering.
        Use the research tool to find information on any topic requested."""
