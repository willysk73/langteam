"""LangGroup - A multiagent system framework built on LangChain and LangGraph.

This package provides a supervisor-based architecture for coordinating
multiple specialized AI agents to work together on complex tasks.
"""

from .agent_system import AgentSystem
from .team_supervisor import TeamSupervisor
from .models import AgentState, RouteDecision
from .agents import BaseAgent, SupervisorAgent

__version__ = "0.2.0"
__all__ = [
    "AgentSystem",
    "TeamSupervisor",
    "AgentState",
    "RouteDecision",
    "BaseAgent",
    "SupervisorAgent",
]
