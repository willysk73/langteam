"""Base class for creating specialized agents."""

import logging
from abc import ABC, abstractmethod
from typing import List, Callable, Optional
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for a specialized agent.

    Agents can have both tools (for task execution) and sub-agents (for coordination).
    When sub-agents are provided, they are automatically converted to callable tools.
    """

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a description of the agent."""
        pass

    def __init__(
        self,
        llm: BaseChatModel,
        name: Optional[str] = None,
        tools: Optional[List[Callable]] = None,
        agents: Optional[List["BaseAgent"]] = None,
        system_prompt: Optional[str] = None,
        verbose: bool = False,
    ):
        """Initialize the agent with a language model and optional configuration.

        Args:
            llm: The language model to use
            name: Optional custom name for the agent. If not provided, uses class name.
            tools: Optional list of tools for the agent to use
            agents: Optional list of sub-agents. When provided, they're converted to tools.
            system_prompt: Optional system prompt for the agent
            verbose: If True, logs agent activity
        """
        self.llm = llm
        self.name = name or self.__class__.__name__
        self._base_tools = tools or []
        self.sub_agents = agents
        self._base_system_prompt = system_prompt or ""
        self.verbose = verbose

        # Track conversation history for this agent
        self.conversation_history = []

        # Create the agent with combined tools (base + agent tools)
        self.agent = self._create_agent()

    @property
    def tools(self) -> List[Callable]:
        """Return a list of tools for the agent.

        Combines base tools with sub-agents converted to callable tools.
        """
        # If we have sub-agents, convert them to tools
        if self.sub_agents:
            agent_tools = [self._create_agent_tool(agent) for agent in self.sub_agents]
            return self._base_tools + agent_tools

        return self._base_tools

    @property
    def system_prompt(self) -> str:
        """Return the system prompt for the agent.

        If sub-agents are provided, augments the base prompt with agent coordination guidance.
        """
        if self.sub_agents:
            agent_guidance = self._generate_agent_guidance()
            if self._base_system_prompt:
                return f"{self._base_system_prompt}\n\n{agent_guidance}"
            return agent_guidance

        return self._base_system_prompt

    def _generate_agent_guidance(self) -> str:
        """Generate guidance for working with sub-agents."""
        agent_descriptions = "\n".join(
            f"- {agent.name}: {agent.description}" for agent in self.sub_agents
        )

        return f"""You have access to specialized sub-agents:

{agent_descriptions}

When delegating to the next agent, include relevant results from previous agents in your instruction."""

    def _create_agent(self):
        """Create and compile the agent graph with tools."""
        # Configure LLM - disable parallel tool calls for sub-agents
        llm = self.llm
        if self.sub_agents:
            # Bind configuration before creating agent
            print("HERERERE")
            llm = llm.bind_tools(tools=self.tools, parallel_tool_calls=False)

        return create_agent(model=llm, tools=self.tools, system_prompt=self.system_prompt)

    def _create_agent_tool(self, agent: "BaseAgent") -> Callable:
        """Convert a sub-agent into a callable tool."""

        def agent_tool(task: str) -> str:
            """Execute task using the sub-agent with parent's conversation history."""
            if self.verbose:
                logger.info(f"\n{'─' * 60}")
                logger.info(f"🔀 {self.name} → {agent.name}")
                logger.info(f"📨 Passing context: {task[:200]}{'...' if len(task) > 200 else ''}")
                if self.conversation_history:
                    logger.info(
                        f"📚 Including {len(self.conversation_history)} previous messages from parent"
                    )
                    # Debug: show what types of messages we're passing
                    for i, msg in enumerate(self.conversation_history):
                        msg_type = type(msg).__name__
                        content_preview = str(msg.content if hasattr(msg, "content") else msg)
                        logger.info(f"  [{i}] {msg_type}: {content_preview}...")
                logger.info(f"{'─' * 60}")

            # Pass parent's conversation history + current task to sub-agent
            from langchain_core.messages import HumanMessage

            messages = self.conversation_history + [HumanMessage(content=task)]
            result = agent.invoke({"messages": messages})

            # Extract the response content
            if isinstance(result, dict) and "messages" in result:
                result_messages = result["messages"]
                if result_messages:
                    last_msg = result_messages[-1]
                    response = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

                    # Update parent's history with the tool call and result
                    # This allows subsequent tool calls to see this interaction
                    from langchain_core.messages import AIMessage, ToolMessage

                    # Add AI message for the tool call
                    tool_call_msg = AIMessage(content=f"Calling {agent.name} with: {task}")
                    tool_result_msg = AIMessage(content=response, name=agent.name)
                    self.conversation_history.extend([tool_call_msg, tool_result_msg])

                    if self.verbose:
                        logger.info(f"\n{'─' * 60}")
                        logger.info(f"🔙 {agent.name} → {self.name}")
                        logger.info(
                            f"📬 Returning result: {response[:200]}{'...' if len(response) > 200 else ''}"
                        )
                        logger.info(
                            f"📚 Updated parent history (now {len(self.conversation_history)} messages)"
                        )
                        logger.info(f"{'─' * 60}")
                    return response

            return str(result)

        # Set tool metadata for LangChain
        agent_tool.__name__ = agent.name
        agent_tool.__doc__ = agent.description

        return agent_tool

    def invoke(self, inputs, **kwargs):
        """Invoke the agent.

        Maintains conversation history and passes it to sub-agents.
        """
        # Convert incoming messages to proper format
        if isinstance(inputs, dict) and "messages" in inputs:
            input_messages = inputs["messages"]
            from langchain_core.messages import HumanMessage

            converted_messages = []
            for msg in input_messages:
                if isinstance(msg, tuple):
                    converted_messages.append(HumanMessage(content=msg[1]))
                else:
                    converted_messages.append(msg)
            # Update conversation history with converted messages
            self.conversation_history = converted_messages

        if self.verbose:
            # Extract task from inputs
            if isinstance(inputs, dict) and "messages" in inputs:
                messages = inputs["messages"]
                if messages:
                    if isinstance(messages[-1], tuple):
                        task = messages[-1][1]
                    else:
                        task = (
                            messages[-1].content
                            if hasattr(messages[-1], "content")
                            else str(messages[-1])
                        )
                    logger.info(f"\n{'=' * 60}")
                    logger.info(f"🤖 {self.name} STARTING")
                    logger.info(f"📝 Task: {task}")
                    logger.info(f"📚 Context: {len(self.conversation_history)} messages in history")
                    logger.info(f"{'=' * 60}")

        result = self.agent.invoke(inputs, **kwargs)

        # Update conversation history with result (includes tool calls, tool results, agent responses)
        if isinstance(result, dict) and "messages" in result:
            self.conversation_history = list(result["messages"])

        if self.verbose:
            # Log completion
            if isinstance(result, dict) and "messages" in result:
                response = result["messages"][-1].content if result["messages"] else "No response"
                logger.info(f"\n{'─' * 60}")
                logger.info(f"✅ {self.name} COMPLETED")
                logger.info(f"📤 Result:\n{response}")
                logger.info(f"{'─' * 60}\n")

        return result
