"""Test script for the multiagent system."""

import os
import sys
import pytest
from dotenv import load_dotenv

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../examples")))

from langgroup import BaseAgent
from langchain_openai import ChatOpenAI
from example_agents import (
    ResearchAgent,
    AnalysisAgent,
    WritingAgent,
    MathAgent,
)


class CoordinatorAgent(BaseAgent):
    """A simple coordinator agent for running multiple sub-agents."""

    @property
    def description(self) -> str:
        return "Coordinates multiple specialized agents to complete complex tasks"

    def __init__(self, llm, **kwargs):
        super().__init__(
            llm,
            system_prompt="You are a coordinator that delegates tasks to specialized agents.",
            **kwargs,
        )


# Load environment variables
load_dotenv()

# Marker for tests that require OpenAI API key
requires_openai = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not found in .env file"
)


@requires_openai
def test_basic_workflow():
    """Test basic multiagent workflow."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    sub_agents = [
        ResearchAgent(llm, verbose=True),
        AnalysisAgent(llm, verbose=True),
        WritingAgent(llm, verbose=True),
        MathAgent(llm, verbose=True),
    ]
    coordinator = CoordinatorAgent(llm, name="Coordinator", agents=sub_agents, verbose=True)
    test_task = "Calculate 15 * 8 and then write a brief summary of the result"
    result = coordinator.invoke({"messages": [("human", test_task)]})

    assert result is not None
    assert "messages" in result
    # Check that the result contains expected content
    final_message = result["messages"][-1].content
    assert "120" in final_message  # 15 * 8 = 120


@requires_openai
def test_research_and_write_workflow():
    """Test a workflow that requires research and writing."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    sub_agents = [
        ResearchAgent(llm),
        AnalysisAgent(llm),
        WritingAgent(llm),
        MathAgent(llm),
    ]
    coordinator = CoordinatorAgent(llm, name="Coordinator", agents=sub_agents)
    test_task = "Research the capital of France and write it down."
    result = coordinator.invoke({"messages": [("human", test_task)]})

    assert result is not None
    assert "messages" in result
    # Check if the final output contains "Paris"
    final_output = result["messages"][-1].content
    assert "Paris" in final_output


@requires_openai
def test_multi_step_math_and_analysis():
    """Test a multi-step calculation and analysis."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    sub_agents = [
        ResearchAgent(llm),
        AnalysisAgent(llm),
        WritingAgent(llm),
        MathAgent(llm),
    ]
    coordinator = CoordinatorAgent(llm, name="Coordinator", agents=sub_agents)
    test_task = "If a population of 1000 grows by 10% each year for 2 years, what is the final population? Analyze the growth."
    result = coordinator.invoke({"messages": [("human", test_task)]})

    assert result is not None
    assert "messages" in result
    # Check for the result of the calculation in the output
    final_output = result["messages"][-1].content
    assert "1210" in final_output


@requires_openai
def test_integral_calculation():
    """Test agent coordination with integral calculation."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    sub_agents = [
        ResearchAgent(llm, verbose=True),
        AnalysisAgent(llm, verbose=True),
        WritingAgent(llm, verbose=True),
        MathAgent(llm, verbose=True),
    ]
    coordinator = CoordinatorAgent(llm, name="Coordinator", agents=sub_agents, verbose=True)
    test_task = "Calculate the integral of x^2 from 0 to 3, then analyze what this integral represents geometrically, and finally write a summary explaining both the calculation and its geometric meaning."
    result = coordinator.invoke({"messages": [("human", test_task)]})

    assert result is not None
    assert "messages" in result
    # Check that the result contains relevant content
    final_output = result["messages"][-1].content
    assert len(final_output) > 0
    # The integral of x^2 from 0 to 3 is 9
    assert "9" in final_output


@requires_openai
def test_hierarchical_supervisors():
    """Test hierarchical supervisor structure with sub-agents under BaseAgent."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Create two teams of specialized agents
    # Team 1: Research and Analysis
    research_team = [
        ResearchAgent(llm),
        AnalysisAgent(llm),
    ]
    research_supervisor = CoordinatorAgent(llm, name="ResearchTeam", agents=research_team)

    # Team 2: Math and Writing
    content_team = [
        MathAgent(llm),
        WritingAgent(llm),
    ]
    content_supervisor = CoordinatorAgent(llm, name="ContentTeam", agents=content_team)

    # Top-level coordinator coordinates the two team coordinators
    top_level_agents = [research_supervisor, content_supervisor]
    top_coordinator = CoordinatorAgent(llm, name="TopCoordinator", agents=top_level_agents)

    # Task that requires coordination between teams
    test_task = "Research the concept of compound interest, calculate 1000 * (1.05)^3, and write a summary explaining the result"
    result = top_coordinator.invoke({"messages": [("human", test_task)]})

    assert result is not None
    assert "messages" in result
    # Check that the task completed successfully with relevant content
    final_output = result["messages"][-1].content
    assert len(final_output) > 0


if __name__ == "__main__":
    # This allows running the tests directly from the script
    # The -s flag shows print statements, -v is for verbose output
    pytest.main([__file__, "-s", "-v"])
