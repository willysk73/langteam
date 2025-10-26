"""Agent definitions for the multiagent system."""
from .research_agent import create_research_agent
from .analysis_agent import create_analysis_agent
from .writing_agent import create_writing_agent
from .math_agent import create_math_agent

__all__ = [
    "create_research_agent",
    "create_analysis_agent",
    "create_writing_agent",
    "create_math_agent",
]
