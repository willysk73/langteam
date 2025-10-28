import os
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain import hub

# Load environment variables
load_dotenv()


def calculator(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        result = eval(expression)
        return f"The result is: {result}"
    except Exception as e:
        return f"Error calculating: {str(e)}"


def get_weather(location: str) -> str:
    """Get weather information for a location (mock implementation)."""
    return f"The weather in {location} is sunny and 72°F."


def main():
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please create a .env file with your OpenAI API key.")
        return

    # Define tools
    tools = [
        Tool(
            name="Calculator",
            func=calculator,
            description="Useful for mathematical calculations. Input should be a valid Python expression.",
        ),
        Tool(
            name="Weather",
            func=get_weather,
            description="Get weather information for a location. Input should be a city name.",
        ),
    ]

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4", temperature=0)

    # Get the ReAct prompt from LangChain hub
    prompt = hub.pull("hwchase17/react")

    # Create agent
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # Example queries
    print("\n=== LangChain Agent Demo ===")
    print("\nQuery 1: What's 25 * 4 + 10?")
    result1 = agent_executor.invoke({"input": "What's 25 * 4 + 10?"})
    print(f"\nAnswer: {result1['output']}")

    print("\n" + "=" * 50)
    print("\nQuery 2: What's the weather in San Francisco?")
    result2 = agent_executor.invoke({"input": "What's the weather in San Francisco?"})
    print(f"\nAnswer: {result2['output']}")


if __name__ == "__main__":
    main()
