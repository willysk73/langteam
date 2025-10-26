"""Analysis agent for data analysis and insights."""
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from tools import analysis_tool


def create_analysis_agent(llm: BaseChatModel):
    """Create an analysis agent for data analysis.
    
    Args:
        llm: The language model to use for the agent
        
    Returns:
        Compiled agent graph configured with analysis tools
    """
    tools = [analysis_tool]
    
    system_prompt = """You are an analysis agent specializing in data analysis and insights.
    Use the analysis tool to analyze data and identify patterns."""
    
    return create_agent(
        llm,
        tools=tools,
        system_prompt=system_prompt
    )
