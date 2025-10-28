"""Example demonstrating verbose logging with agents."""

import logging
from langchain_openai import ChatOpenAI
from example_agents import MathAgent, WritingAgent, ResearchAgent
from langgroup import BaseAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%H:%M:%S")


class CoordinatorAgent(BaseAgent):
    """A simple coordinator agent."""

    @property
    def description(self) -> str:
        return "Coordinates multiple specialized agents to complete complex tasks"

    def __init__(self, llm, **kwargs):
        super().__init__(
            llm,
            system_prompt="You are a coordinator that delegates tasks to specialized agents.",
            **kwargs,
        )


def main():
    """Run example with verbose logging."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Create agents with verbose=True
    sub_agents = [
        MathAgent(llm, verbose=True),
        WritingAgent(llm, verbose=True),
        ResearchAgent(llm, verbose=True),
    ]

    coordinator = CoordinatorAgent(llm, name="Coordinator", agents=sub_agents, verbose=True)

    print("\n" + "=" * 60)
    print("Running agent with verbose logging enabled")
    print("=" * 60 + "\n")

    task = "Calculate 25 * 4 and write a brief summary of the result"
    result = coordinator.invoke({"messages": [("human", task)]})

    print("\n" + "=" * 60)
    print("Final result:")
    print("=" * 60)
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
