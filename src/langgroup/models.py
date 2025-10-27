"""Data models for the multiagent system."""
from typing import Literal, TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage




class RouteDecision(BaseModel):
    """Decision made by the supervisor about which agent to route to next."""
    next_agent: str = Field(
        description="The name of the agent to handle the next step, or 'finish' if task is complete"
    )
    reasoning: str = Field(description="Brief explanation of why this agent was chosen")


class AgentState(TypedDict):
    """State that will be passed between agents in the workflow."""
    messages: list[BaseMessage]
    next: str
    task_result: dict
