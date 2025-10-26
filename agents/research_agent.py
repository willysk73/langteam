"""Research agent for information gathering."""
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain import hub
from tools import research_tool


def create_research_agent(llm: ChatOpenAI) -> AgentExecutor:
    """Create a research agent for information gathering.
    
    Args:
        llm: The language model to use for the agent
        
    Returns:
        AgentExecutor configured with research tools
    """
    tools = [
        Tool(
            name="Research",
            func=research_tool,
            description="Research information on any topic. Use this for information gathering tasks."
        )
    ]
    
    prompt = hub.pull("hwchase17/openai-functions-agent")
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    return AgentExecutor(agent=agent, tools=tools, verbose=True)
