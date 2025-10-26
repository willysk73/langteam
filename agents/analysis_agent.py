"""Analysis agent for data analysis and insights."""
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool
from langchain_core.language_models import BaseChatModel
from langchain import hub
from tools import analysis_tool


def create_analysis_agent(llm: BaseChatModel) -> AgentExecutor:
    """Create an analysis agent for data analysis.
    
    Args:
        llm: The language model to use for the agent
        
    Returns:
        AgentExecutor configured with analysis tools
    """
    tools = [
        Tool(
            name="Analyze",
            func=analysis_tool,
            description="Analyze data and provide insights. Use this for analytical tasks."
        )
    ]
    
    prompt = hub.pull("hwchase17/openai-functions-agent")
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    return AgentExecutor(agent=agent, tools=tools, verbose=True)
