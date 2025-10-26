import os
from typing import Annotated, Literal, TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langchain import hub
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()


# Define agent metadata for dynamic routing
class AgentInfo(BaseModel):
    """Information about an available agent."""
    name: str = Field(description="The agent's identifier name")
    description: str = Field(description="What this agent specializes in")


# Pydantic model for supervisor routing decision
class RouteDecision(BaseModel):
    """Decision made by the supervisor about which agent to route to next."""
    next_agent: Literal["research_agent", "analysis_agent", "writing_agent", "math_agent", "FINISH"] = Field(
        description="The name of the agent to handle the next step, or FINISH if task is complete"
    )
    reasoning: str = Field(description="Brief explanation of why this agent was chosen")


# Define the state that will be passed between agents
class AgentState(TypedDict):
    messages: list[BaseMessage]
    next: str
    task_result: dict


# Define specialized agent tools
def research_tool(query: str) -> str:
    """Research information on a given topic."""
    return f"Research findings for '{query}': This is simulated research data with key insights and facts."


def analysis_tool(data: str) -> str:
    """Analyze data and provide insights."""
    return f"Analysis of '{data}': Key patterns identified - trend A, correlation B, and insight C."


def writing_tool(content: str) -> str:
    """Write and format content based on input."""
    return f"Formatted content: {content}\n\nThis has been professionally formatted and structured."


def calculation_tool(expression: str) -> str:
    """Perform mathematical calculations."""
    try:
        result = eval(expression)
        return f"Calculation result: {result}"
    except Exception as e:
        return f"Error in calculation: {str(e)}"


# Create specialized sub-agents
class MultiAgentSystem:
    def __init__(self):
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        self.supervisor_llm = ChatOpenAI(model="gpt-4", temperature=0)
        
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
        
        # Create sub-agents with specialized tools
        self.research_agent = self._create_agent(
            "Research Agent",
            [Tool(name="Research", func=research_tool, 
                  description="Research information on any topic")]
        )
        
        self.analysis_agent = self._create_agent(
            "Analysis Agent",
            [Tool(name="Analyze", func=analysis_tool,
                  description="Analyze data and provide insights")]
        )
        
        self.writing_agent = self._create_agent(
            "Writing Agent",
            [Tool(name="Write", func=writing_tool,
                  description="Write and format content")]
        )
        
        self.math_agent = self._create_agent(
            "Math Agent",
            [Tool(name="Calculate", func=calculation_tool,
                  description="Perform mathematical calculations")]
        )
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
    
    def _create_agent(self, name: str, tools: list[Tool]) -> AgentExecutor:
        """Create a specialized agent with given tools."""
        prompt = hub.pull("hwchase17/openai-functions-agent")
        agent = create_openai_functions_agent(self.llm, tools, prompt)
        return AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    def _supervisor_node(self, state: AgentState) -> AgentState:
        """Supervisor decides which agent should act next using structured output."""
        messages = state["messages"]
        
        # Build agent descriptions dynamically
        agent_descriptions = "\n".join(
            f"- {agent.name}: {agent.description}" 
            for agent in self.available_agents
        )
        
        # Create supervisor prompt with structured output
        supervisor_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a supervisor coordinating a team of specialized agents.

Available agents:
{agent_descriptions}

Your job is to analyze the current task and conversation history, then decide which agent 
should handle the next step. Choose FINISH when the task is complete.

Consider:
- What has already been done by previous agents
- What remains to be done
- Which agent is best suited for the next step"""),
            ("human", """Task: {task}

Conversation history:
{history}

Decide which agent should act next or if we should FINISH.""")
        ])
        
        # Use structured output with Pydantic
        structured_llm = self.supervisor_llm.with_structured_output(RouteDecision)
        
        # Build conversation history
        history = "\n".join([msg.content for msg in messages])
        task = messages[0].content if messages else "No task"
        
        # Get structured decision
        decision = structured_llm.invoke(
            supervisor_prompt.format_messages(
                task=task,
                history=history
            )
        )
        
        next_agent = decision.next_agent.lower()
        
        print(f"\n🎯 Supervisor decision: {next_agent}")
        print(f"💭 Reasoning: {decision.reasoning}")
        
        return {"messages": messages, "next": next_agent, "task_result": state.get("task_result", {})}
    
    def _agent_node(self, agent: AgentExecutor, agent_name: str):
        """Create a node for a specific agent."""
        def node(state: AgentState) -> AgentState:
            messages = state["messages"]
            last_message = messages[-1].content if messages else ""
            
            print(f"\n🤖 {agent_name} is working...")
            result = agent.invoke({"input": last_message})
            
            # Add agent's response to messages
            new_message = HumanMessage(
                content=f"{agent_name} result: {result['output']}",
                name=agent_name
            )
            
            return {
                "messages": messages + [new_message],
                "next": "",
                "task_result": {**state.get("task_result", {}), agent_name: result['output']}
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
