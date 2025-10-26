"""Writing agent for content creation and formatting."""
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from tools import writing_tool


def create_writing_agent(llm: BaseChatModel):
    """Create a writing agent for content creation.
    
    Args:
        llm: The language model to use for the agent
        
    Returns:
        Compiled agent graph configured with writing tools
    """
    tools = [writing_tool]
    
    system_prompt = """You are a writing agent specializing in content creation and formatting.
    Use the writing tool to create and format professional content."""
    
    return create_agent(
        llm,
        tools=tools,
        system_prompt=system_prompt
    )
