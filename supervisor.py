"""Supervisor agent for coordinating sub-agents."""
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from models import AgentInfo, AgentState, RouteDecision


class Supervisor:
    """Supervisor agent that routes tasks to specialized sub-agents."""
    
    def __init__(self, llm: ChatOpenAI, available_agents: list[AgentInfo]):
        """Initialize the supervisor.
        
        Args:
            llm: The language model to use for decision making
            available_agents: List of available agents with their metadata
        """
        self.llm = llm
        self.available_agents = available_agents
        self.structured_llm = llm.with_structured_output(RouteDecision)
    
    def decide_next_agent(self, state: AgentState) -> AgentState:
        """Decide which agent should act next based on current state.
        
        Args:
            state: Current agent state containing messages and results
            
        Returns:
            Updated state with the next agent decision
        """
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
        
        # Build conversation history
        history = "\n".join([msg.content for msg in messages])
        task = messages[0].content if messages else "No task"
        
        # Get structured decision
        decision = self.structured_llm.invoke(
            supervisor_prompt.format_messages(
                task=task,
                history=history
            )
        )
        
        next_agent = decision.next_agent.lower()
        
        print(f"\n🎯 Supervisor decision: {next_agent}")
        print(f"💭 Reasoning: {decision.reasoning}")
        
        return {
            "messages": messages, 
            "next": next_agent, 
            "task_result": state.get("task_result", {})
        }
