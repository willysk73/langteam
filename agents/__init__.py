"""Agent definitions for the multiagent system."""
from .research_agent import ResearchAgent
from .analysis_agent import AnalysisAgent
from .writing_agent import WritingAgent
from .math_agent import MathAgent
from .supervisor_agent import SupervisorAgent

__all__ = ["ResearchAgent", "AnalysisAgent", "WritingAgent", "MathAgent", "SupervisorAgent"]
