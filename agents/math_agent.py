"""Math agent for calculations and computations."""
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from tools import calculation_tool


def create_math_agent(llm: BaseChatModel):
    """Create a math agent for calculations.
    
    Args:
        llm: The language model to use for the agent
        
    Returns:
        Compiled agent graph configured with calculation tools
    """
    tools = [calculation_tool]
    
    system_prompt = """You are a math agent specializing in calculations and computations.
    Use the calculation tool to perform mathematical operations."""
    
    return create_agent(
        llm,
        tools=tools,
        system_prompt=system_prompt
    )
