"""LangGroup - A multiagent system framework built on LangChain and LangGraph.

This package provides a flexible architecture where agents can have both tools
and sub-agents, enabling hierarchical agent coordination.
"""

from .models import AgentState, RouteDecision
from .agents import BaseAgent

__version__ = "0.3.0"
__all__ = [
    "AgentState",
    "RouteDecision",
    "BaseAgent",
]
