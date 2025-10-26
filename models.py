"""Data models for the multiagent system."""
from typing import Literal, TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage


class AgentInfo(BaseModel):
    """Information about an available agent."""
    name: str = Field(description="The agent's identifier name")
    description: str = Field(description="What this agent specializes in")


class RouteDecision(BaseModel):
    """Decision made by the supervisor about which agent to route to next."""
    next_agent: Literal["research_agent", "analysis_agent", "writing_agent", "math_agent", "FINISH"] = Field(
        description="The name of the agent to handle the next step, or FINISH if task is complete"
    )
    reasoning: str = Field(description="Brief explanation of why this agent was chosen")


class AgentState(TypedDict):
    """State that will be passed between agents in the workflow."""
    messages: list[BaseMessage]
    next: str
    task_result: dict
