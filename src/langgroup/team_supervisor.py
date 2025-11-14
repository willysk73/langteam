"""Supervisor agent for coordinating sub-agents."""
import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from .models import AgentState, RouteDecision
from .agents.base_agent import BaseAgent
from .agents.supervisor_agent import SupervisorAgent

logger = logging.getLogger(__name__)


class TeamSupervisor:
    """Supervisor agent that routes tasks to specialized sub-agents."""
    
    def __init__(self, llm: BaseChatModel, available_agents: list[BaseAgent]):
        """Initialize the supervisor.
        
        Args:
            llm: The language model to use for decision making
            available_agents: List of available agents with their metadata
        """
        self.llm = llm
        self.available_agents = available_agents
        self.supervisor_agent = SupervisorAgent(llm, available_agents)
        self.structured_llm = llm.with_structured_output(RouteDecision)
    
    def decide_next_agent(self, state: AgentState) -> AgentState:
        """Decide which agent should act next based on current state.
        
        Args:
            state: Current agent state containing messages and results
            
        Returns:
            Updated state with the next agent decision
        """
        messages = state["messages"]
        
        # Build agent descriptions dynamically with agent names
        agent_descriptions = "\n".join(
            f"- {agent.name}: {agent.description}" 
            for agent in self.available_agents
        )
        
        # Create supervisor prompt with structured output
        system_prompt = f"""You are a supervisor orchestrating a team of specialized agents. Your role is to analyze the user's task and the ongoing conversation to delegate the next step to the most appropriate agent.

Here are the agents available to you:
{agent_descriptions}

Follow these rules:
1.  **Analyze the Request**: Carefully read the user's task and the conversation history.
2.  **Break Down the Task**: If the task is complex, break it down into smaller, sequential steps. Each step should be handled by the most appropriate agent.
3.  **Delegate**: Choose the best agent to perform the next action using their exact name from the list above.
4.  **FINISH**: Once all steps of the task are fully completed and the user's request has been met, you must respond with "finish". Do not finish if there are still steps to be done.
5.  **No Assumptions**: Do not make assumptions about what has been done. Base your decisions only on the conversation history. If the history is empty, start from the beginning of the task.

Your job is to decide which agent should act next by returning their exact name, or "finish" when the entire task is complete."""
        
        supervisor_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
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
        
        next_agent = decision.next_agent
        
        logger.info(f"🎯 Supervisor decision: {next_agent}")
        logger.info(f"💭 Reasoning: {decision.reasoning}")
        
        return {
            "messages": messages, 
            "next": next_agent, 
            "task_result": state.get("task_result", {})
        }
