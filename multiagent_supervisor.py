import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

from models import AgentInfo, AgentState
from team_supervisor import TeamSupervisor
from agents import (
    ResearchAgent,
    AnalysisAgent,
    WritingAgent,
    MathAgent,
)

# Load environment variables
load_dotenv()


class MultiAgentSystem:
    """Multiagent system with supervisor coordination."""
    
    def __init__(self):
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # Define available agents with metadata
        self.available_agents = [
            AgentInfo(
                name="research_agent",
                description="Researches information and gathers data on any topic. Use for information gathering tasks."
            ),
            AgentInfo(
                name="analysis_agent",
                description="Analyzes data, identifies patterns, and provides insights. Use for analytical tasks."
            ),
            AgentInfo(
                name="writing_agent",
                description="Writes, formats, and structures content professionally. Use for content creation tasks."
            ),
            AgentInfo(
                name="math_agent",
                description="Performs mathematical calculations and computations. Use for numerical tasks."
            ),
        ]
        
        # Initialize supervisor
        self.supervisor = TeamSupervisor(self.llm, self.available_agents)
        
        # Create sub-agents
        self.research_agent = ResearchAgent(self.llm)
        self.analysis_agent = AnalysisAgent(self.llm)
        self.writing_agent = WritingAgent(self.llm)
        self.math_agent = MathAgent(self.llm)

        # Build the workflow graph
        self.workflow = self._build_workflow()

    def _supervisor_node(self, state: AgentState) -> AgentState:
        """Supervisor node that delegates to the Supervisor class."""
        return self.supervisor.decide_next_agent(state)

    def _agent_node(self, agent, agent_name: str):
        """Create a node for a specific agent."""
        def node(state: AgentState) -> AgentState:
            messages = state["messages"]
            last_message = messages[-1].content if messages else ""
            
            print(f"\n🤖 {agent_name} is working...")
            # The new create_agent returns a compiled graph, which is invoked directly
            result = agent.invoke({"messages": [("human", last_message)]})
            
            # Extract the agent's response from the result
            agent_response = result['messages'][-1].content

            # Add agent's response to messages
            new_message = HumanMessage(
                content=f"{agent_name} result: {agent_response}",
                name=agent_name
            )
            
            return {
                "messages": messages + [new_message],
                "next": "",
                "task_result": {**state.get("task_result", {}), agent_name: agent_response}
            }
        
        return node
    
    def _build_workflow(self) -> StateGraph:
        """Build the workflow graph with supervisor and agents."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("supervisor", self._supervisor_node)
        workflow.add_node("research_agent", self._agent_node(self.research_agent, "Research Agent"))
        workflow.add_node("analysis_agent", self._agent_node(self.analysis_agent, "Analysis Agent"))
        workflow.add_node("writing_agent", self._agent_node(self.writing_agent, "Writing Agent"))
        workflow.add_node("math_agent", self._agent_node(self.math_agent, "Math Agent"))
        
        # Set entry point
        workflow.set_entry_point("supervisor")
        
        # Add conditional edges from supervisor to agents
        def route_supervisor(state: AgentState) -> str:
            next_agent = state["next"]
            if next_agent == "finish":
                return "end"
            return next_agent
        
        workflow.add_conditional_edges(
            "supervisor",
            route_supervisor,
            {
                "research_agent": "research_agent",
                "analysis_agent": "analysis_agent",
                "writing_agent": "writing_agent",
                "math_agent": "math_agent",
                "end": END
            }
        )
        
        # All agents return to supervisor for next decision
        workflow.add_edge("research_agent", "supervisor")
        workflow.add_edge("analysis_agent", "supervisor")
        workflow.add_edge("writing_agent", "supervisor")
        workflow.add_edge("math_agent", "supervisor")
        
        return workflow.compile()
    
    def run(self, task: str) -> dict:
        """Run the multiagent system with a given task."""
        print(f"\n{'='*60}")
        print(f"🚀 Starting multiagent system")
        print(f"📝 Task: {task}")
        print(f"{'='*60}")
        
        initial_state = {
            "messages": [HumanMessage(content=task)],
            "next": "",
            "task_result": {}
        }
        
        result = self.workflow.invoke(initial_state)
        
        print(f"\n{'='*60}")
        print(f"✅ Task completed!")
        print(f"{'='*60}")
        
        return result


def main():
    """Main function demonstrating the multiagent system."""
    
    # Initialize the multiagent system
    system = MultiAgentSystem()
    
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
