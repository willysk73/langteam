import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgroup import AgentSystem
from example_agents import (
    ResearchAgent,
    AnalysisAgent,
    WritingAgent,
    MathAgent,
)

# Load environment variables
load_dotenv()


def main():
    """Main function demonstrating the multiagent system."""
    
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Create agent instances
    agents = [
        ResearchAgent(llm),
        AnalysisAgent(llm),
        WritingAgent(llm),
        MathAgent(llm),
    ]

    # Initialize the agent system
    system = AgentSystem(llm, agents)
    
    # Example tasks demonstrating agent collaboration
    tasks = [
        "Research artificial intelligence trends and calculate the growth rate if AI adoption increased by 25% per year for 3 years starting at 40%",
        "Analyze the benefits of multiagent systems and write a summary",
    ]
    
    for i, task in enumerate(tasks, 1):
        print(f"\n\n{'#'*60}")
        print(f"TASK {i}")
        print(f"{'#'*60}")
        result = system.run(task)
        
        print(f"\n📊 Final Results:")
        for agent_name, output in result.get("task_result", {}).items():
            print(f"\n{agent_name}:")
            print(f"  {output}")


if __name__ == "__main__":
    main()
