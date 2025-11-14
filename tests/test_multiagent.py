"""Test script for the multiagent system."""
import os
import sys
import pytest
from dotenv import load_dotenv

# Add project root to Python path
sys.path.append("examples")
sys.path.append("src")

from langgroup import AgentSystem, SupervisorAgent
from langchain_openai import ChatOpenAI
from examples.example_agents import (
    ResearchAgent,
    AnalysisAgent,
    WritingAgent,
    MathAgent,
)


# Load environment variables
load_dotenv()

# Marker for tests that require OpenAI API key
requires_openai = pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not found in .env file")

@requires_openai
def test_basic_workflow():
    """Test basic multiagent workflow."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agents = [
        ResearchAgent(llm),
        AnalysisAgent(llm),
        WritingAgent(llm),
        MathAgent(llm),
    ]
    system = AgentSystem(llm, agents)
    test_task = "Calculate 15 * 8 and then write a brief summary of the result"
    result = system.run(test_task)
    
    assert result is not None
    assert "task_result" in result
    # Check that at least math_agent and writing_agent were used
    assert "MathAgent" in result["task_result"]
    assert "WritingAgent" in result["task_result"]

@requires_openai
def test_research_and_write_workflow():
    """Test a workflow that requires research and writing."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agents = [
        ResearchAgent(llm),
        AnalysisAgent(llm),
        WritingAgent(llm),
        MathAgent(llm),
    ]
    system = AgentSystem(llm, agents)
    test_task = "Research the capital of France and write it down."
    result = system.run(test_task)

    assert result is not None
    assert "task_result" in result
    assert "ResearchAgent" in result["task_result"]
    assert "WritingAgent" in result["task_result"]
    # Check if the final output contains "Paris"
    final_output = result['messages'][-1].content
    assert "Paris" in final_output

@requires_openai
def test_multi_step_math_and_analysis():
    """Test a multi-step calculation and analysis."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agents = [
        ResearchAgent(llm),
        AnalysisAgent(llm),
        WritingAgent(llm),
        MathAgent(llm),
    ]
    system = AgentSystem(llm, agents)
    test_task = "If a population of 1000 grows by 10% each year for 2 years, what is the final population? Analyze the growth."
    result = system.run(test_task)

    assert result is not None
    assert "task_result" in result
    assert "MathAgent" in result["task_result"]
    # Check for the result of the calculation
    math_agent_output = result["task_result"]["MathAgent"]
    assert "1210" in math_agent_output

@requires_openai
def test_hierarchical_supervisors():
    """Test hierarchical supervisor structure with supervisor agents under a top supervisor."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Create two teams of specialized agents
    # Team 1: Research and Analysis
    research_team = [
        ResearchAgent(llm),
        AnalysisAgent(llm),
    ]
    research_supervisor = SupervisorAgent(llm, research_team, name="ResearchTeamSupervisor")
    
    # Team 2: Math and Writing
    content_team = [
        MathAgent(llm),
        WritingAgent(llm),
    ]
    content_supervisor = SupervisorAgent(llm, content_team, name="ContentTeamSupervisor")
    
    # Top-level supervisor coordinates the two sub-supervisors
    top_level_agents = [research_supervisor, content_supervisor]
    system = AgentSystem(llm, top_level_agents)
    
    # Task that requires coordination between teams
    test_task = "Research the concept of compound interest, calculate 1000 * (1.05)^3, and write a summary explaining the result"
    result = system.run(test_task)
    
    assert result is not None
    assert "task_result" in result
    # Check that both supervisors were involved
    assert "ResearchTeamSupervisor" in result["task_result"]
    assert "ContentTeamSupervisor" in result["task_result"]
    # The task should complete successfully
    assert result["next"] == "finish" or len(result["task_result"]) > 0

if __name__ == "__main__":
    # This allows running the tests directly from the script
    # The -s flag shows print statements, -v is for verbose output
    pytest.main([__file__, "-s", "-v"])
