
from typing import List, Callable
from .base_agent import BaseAgent
from tools import writing_tool


class WritingAgent(BaseAgent):
    """Agent specializing in content creation."""

    @property
    def description(self) -> str:
        """Return a description of the agent."""
        return "Writes, formats, and structures content professionally. Use for content creation tasks."
    @property
    def tools(self) -> List[Callable]:
        """Return a list of tools for the agent."""
        return [writing_tool]

    @property
    def system_prompt(self) -> str:
        """Return the system prompt for the agent."""
        return """You are a writing agent specializing in content creation and formatting.
        Use the writing tool to create and format professional content."""
