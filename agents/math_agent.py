
from typing import List, Callable
from .base_agent import BaseAgent
from tools import calculation_tool


class MathAgent(BaseAgent):
    """Agent specializing in calculations."""

    @property
    def description(self) -> str:
        """Return a description of the agent."""
        return "Performs mathematical calculations and computations. Use for numerical tasks."
    @property
    def tools(self) -> List[Callable]:
        """Return a list of tools for the agent."""
        return [calculation_tool]

    @property
    def system_prompt(self) -> str:
        """Return the system prompt for the agent."""
        return """You are a math agent specializing in calculations and computations.
        Use the calculation tool to perform mathematical operations."""
