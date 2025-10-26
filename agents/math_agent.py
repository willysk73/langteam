"""Math agent for calculations and computations."""
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool
from langchain_core.language_models import BaseChatModel
from langchain import hub
from tools import calculation_tool


def create_math_agent(llm: BaseChatModel) -> AgentExecutor:
    """Create a math agent for calculations.
    
    Args:
        llm: The language model to use for the agent
        
    Returns:
        AgentExecutor configured with calculation tools
    """
    tools = [
        Tool(
            name="Calculate",
            func=calculation_tool,
            description="Perform mathematical calculations. Use this for numerical tasks."
        )
    ]
    
    prompt = hub.pull("hwchase17/openai-functions-agent")
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    return AgentExecutor(agent=agent, tools=tools, verbose=True)
