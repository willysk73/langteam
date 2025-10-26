"""Test script for the multiagent system."""
import os
import sys
import pytest
from dotenv import load_dotenv

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from multiagent_supervisor import MultiAgentSystem


# Load environment variables
load_dotenv()

# Marker for tests that require OpenAI API key
requires_openai = pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not found in .env file")

@requires_openai
def test_basic_workflow():
    """Test basic multiagent workflow."""
    system = MultiAgentSystem()
    test_task = "Calculate 15 * 8 and then write a brief summary of the result"
    result = system.run(test_task)
    
    assert result is not None
    assert "task_result" in result
    # Check that at least math_agent and writing_agent were used
    assert "Math Agent" in result["task_result"]
    assert "Writing Agent" in result["task_result"]

@requires_openai
def test_research_and_write_workflow():
    """Test a workflow that requires research and writing."""
    system = MultiAgentSystem()
    test_task = "Research the capital of France and write it down."
    result = system.run(test_task)

    assert result is not None
    assert "task_result" in result
    assert "Research Agent" in result["task_result"]
    assert "Writing Agent" in result["task_result"]
    # Check if the final output contains "Paris"
    final_output = result['messages'][-1].content
    assert "Paris" in final_output

@requires_openai
def test_multi_step_math_and_analysis():
    """Test a multi-step calculation and analysis."""
    system = MultiAgentSystem()
    test_task = "If a population of 1000 grows by 10% each year for 2 years, what is the final population? Analyze the growth."
    result = system.run(test_task)

    assert result is not None
    assert "task_result" in result
    assert "Math Agent" in result["task_result"]
    # Check for the result of the calculation
    math_agent_output = result["task_result"]["Math Agent"]
    assert "1210" in math_agent_output

if __name__ == "__main__":
    # This allows running the tests directly from the script
    # The -s flag shows print statements, -v is for verbose output
    pytest.main([__file__, "-s", "-v"])
