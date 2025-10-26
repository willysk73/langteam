"""Analysis agent for data analysis and insights."""
from typing import List, Callable
from .base_agent import BaseAgent
from tools import analysis_tool


class AnalysisAgent(BaseAgent):
    """Agent specializing in data analysis."""

    @property
    def tools(self) -> List[Callable]:
        """Return a list of tools for the agent."""
        return [analysis_tool]

    @property
    def system_prompt(self) -> str:
        """Return the system prompt for the agent."""
        return """You are an analysis agent specializing in data analysis and insights.
        Use the analysis tool to analyze data and identify patterns."""
