"""Agent system for managing and coordinating a team of specialized agents."""
import logging

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from .models import AgentState
from .team_supervisor import TeamSupervisor

logger = logging.getLogger(__name__)

class AgentSystem:
    """Multiagent system with supervisor coordination."""
    
    def __init__(self, llm, agents):
        """Initialize the agent system."""
        self.llm = llm
        self.agents = agents
        self.supervisor = TeamSupervisor(llm, agents)
        self.workflow = self._build_workflow()

    def _supervisor_node(self, state: AgentState) -> AgentState:
        """Supervisor node that delegates to the Supervisor class."""
        return self.supervisor.decide_next_agent(state)

    def _agent_node(self, agent, agent_name: str):
        """Create a node for a specific agent."""
        def node(state: AgentState) -> AgentState:
            messages = state["messages"]
            last_message = messages[-1].content if messages else ""
            
            logger.info(f"🤖 {agent_name} is working...")
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
        
        # Add supervisor node
        workflow.add_node("supervisor", self._supervisor_node)
        
        # Add agent nodes and edges
        # Create a mapping of agent names to node names for routing
        self.agent_name_map = {}
        for agent in self.agents:
            # Use agent.name for node identification
            node_name = agent.name.replace('Agent', '').lower() + '_agent'
            self.agent_name_map[agent.name] = node_name
            workflow.add_node(node_name, self._agent_node(agent, agent.name))
            workflow.add_edge(node_name, "supervisor")
        
        # Set entry point
        workflow.set_entry_point("supervisor")
        
        # Add conditional edges from supervisor to agents
        def route_supervisor(state: AgentState) -> str:
            next_agent = state["next"]
            if next_agent == "finish":
                return "end"
            # Map agent name to node name
            return self.agent_name_map.get(next_agent, next_agent)
        
        conditional_map = {node_name: node_name for node_name in self.agent_name_map.values()}
        conditional_map["end"] = END
        
        workflow.add_conditional_edges(
            "supervisor",
            route_supervisor,
            conditional_map
        )
        
        return workflow.compile()
    
    def run(self, task: str) -> dict:
        """Run the multiagent system with a given task."""
        logger.info(f"🚀 Starting multiagent system")
        logger.info(f"📝 Task: {task}")
        
        initial_state = {
            "messages": [HumanMessage(content=task)],
            "next": "",
            "task_result": {}
        }
        
        result = self.workflow.invoke(initial_state)
        logger.info(f"✅ Task completed!")
        
        return result
