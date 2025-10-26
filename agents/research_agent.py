"""Research agent for information gathering."""
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from tools import research_tool


def create_research_agent(llm: BaseChatModel):
    """Create a research agent for information gathering.
    
    Args:
        llm: The language model to use for the agent
        
    Returns:
        Compiled agent graph configured with research tools
    """
    tools = [research_tool]
    
    system_prompt = """You are a research agent specializing in information gathering.
    Use the research tool to find information on any topic requested."""
    
    return create_agent(
        llm,
        tools=tools,
        system_prompt=system_prompt
    )
