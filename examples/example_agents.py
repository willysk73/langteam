"""Example agent implementations demonstrating how to use the langteam framework."""

from typing import List, Callable
from langgroup.agents import BaseAgent


def research_tool(query: str) -> str:
    """Research information on a given topic."""
    return f"Research findings for '{query}': This is simulated research data with key insights and facts."


def analysis_tool(data: str) -> str:
    """Analyze data and provide insights."""
    return f"Analysis of '{data}': Key patterns identified - trend A, correlation B, and insight C."


def writing_tool(content: str) -> str:
    """Write and format content based on input."""
    return f"Formatted content: {content}\n\nThis has been professionally formatted and structured."


def calculation_tool(expression: str) -> str:
    """Perform mathematical calculations."""
    try:
        result = eval(expression)
        return f"Calculation result: {result}"
    except Exception as e:
        return f"Error in calculation: {str(e)}"


class ResearchAgent(BaseAgent):
    """Agent specializing in information gathering."""

    @property
    def description(self) -> str:
        """Return a description of the agent."""
        return "Researches information and gathers data on any topic. Use for information gathering tasks."
    
    @property
    def tools(self) -> List[Callable]:
        """Return a list of tools for the agent."""
        return [research_tool]

    @property
    def system_prompt(self) -> str:
        """Return the system prompt for the agent."""
        return """You are a research agent specializing in information gathering.
        Use the research tool to find information on any topic requested."""


class AnalysisAgent(BaseAgent):
    """Agent specializing in data analysis."""

    @property
    def description(self) -> str:
        """Return a description of the agent."""
        return "Analyzes data and identifies patterns and insights. Use for data analysis tasks."
    
    @property
    def tools(self) -> List[Callable]:
        """Return a list of tools for the agent."""
        return [analysis_tool]

    @property
    def system_prompt(self) -> str:
        """Return the system prompt for the agent."""
        return """You are an analysis agent specializing in data analysis.
        Use the analysis tool to examine data and provide insights."""


class WritingAgent(BaseAgent):
    """Agent specializing in writing and formatting content."""

    @property
    def description(self) -> str:
        """Return a description of the agent."""
        return "Writes and formats content professionally. Use for writing and documentation tasks."
    
    @property
    def tools(self) -> List[Callable]:
        """Return a list of tools for the agent."""
        return [writing_tool]

    @property
    def system_prompt(self) -> str:
        """Return the system prompt for the agent."""
        return """You are a writing agent specializing in content creation.
        Use the writing tool to format and present information clearly."""


class MathAgent(BaseAgent):
    """Agent specializing in mathematical calculations."""

    @property
    def description(self) -> str:
        """Return a description of the agent."""
        return "Performs mathematical calculations and computations. Use for math and calculation tasks."
    
    @property
    def tools(self) -> List[Callable]:
        """Return a list of tools for the agent."""
        return [calculation_tool]

    @property
    def system_prompt(self) -> str:
        """Return the system prompt for the agent."""
        return """You are a math agent specializing in calculations.
        Use the calculation tool to solve mathematical problems."""
