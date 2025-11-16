"""Demo showing tool call logging with arguments and return values."""

import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgroup.agents import BaseAgent
from langgroup.agent_system import AgentSystem
from typing import List, Callable

# Load environment variables
load_dotenv()

# Configure logging to show tool calls
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def add_numbers(a: int, b: int) -> str:
    """Add two numbers together."""
    result = a + b
    return f"The sum of {a} and {b} is {result}"


def multiply_numbers(a: int, b: int) -> str:
    """Multiply two numbers together."""
    result = a * b
    return f"The product of {a} and {b} is {result}"


class CalculatorAgent(BaseAgent):
    """Agent that performs basic calculations."""

    @property
    def description(self) -> str:
        return "Performs basic mathematical calculations like addition and multiplication."
    
    @property
    def tools(self) -> List[Callable]:
        return [add_numbers, multiply_numbers]

    @property
    def system_prompt(self) -> str:
        return """You are a calculator agent. Use the tools to perform calculations.
        When given a task, use the appropriate tool and return the result."""


def main():
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please create a .env file with your OpenAI API key.")
        return

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Create agent
    calc_agent = CalculatorAgent(llm=llm)

    # Create agent system
    system = AgentSystem(
        agents=[calc_agent],
        llm=llm
    )

    # Run a task that will trigger tool calls
    print("\n=== Tool Logging Demo ===\n")
    print("Task: Calculate 15 + 27, then multiply the result by 3\n")
    print("Watch the logs below to see arguments and return values:\n")
    
    result = system.run("Calculate 15 + 27, then multiply the result by 3")
    
    print("\n=== Final Result ===")
    print(result["task_result"])


if __name__ == "__main__":
    main()
