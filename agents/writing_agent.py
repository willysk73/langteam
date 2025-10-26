"""Writing agent for content creation and formatting."""
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain import hub
from tools import writing_tool


def create_writing_agent(llm: ChatOpenAI) -> AgentExecutor:
    """Create a writing agent for content creation.
    
    Args:
        llm: The language model to use for the agent
        
    Returns:
        AgentExecutor configured with writing tools
    """
    tools = [
        Tool(
            name="Write",
            func=writing_tool,
            description="Write and format content professionally. Use this for content creation tasks."
        )
    ]
    
    prompt = hub.pull("hwchase17/openai-functions-agent")
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    return AgentExecutor(agent=agent, tools=tools, verbose=True)
