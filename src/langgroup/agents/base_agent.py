
"""Base class for creating specialized agents."""
from abc import ABC, abstractmethod
from typing import List, Callable, Optional
import logging
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from typing import Any, Dict


logger = logging.getLogger(__name__)


class ToolCallLogger(BaseCallbackHandler):
    """Callback handler to log tool calls."""

    def __init__(self, agent_name: str):
        self.agent_name = agent_name

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Log when a tool starts execution."""
        tool_name = serialized.get("name", "Unknown")
        logger.info(f"[{self.agent_name}] Calling tool: {tool_name}")
        logger.info(f"[{self.agent_name}] Arguments: {input_str}")

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Log when a tool finishes execution."""
        logger.info(f"[{self.agent_name}] Tool completed")
        # Extract content if output is a ToolMessage object, otherwise use as-is
        return_value = output.content if hasattr(output, 'content') else output
        logger.info(f"[{self.agent_name}] Return value: {return_value}")


class BaseAgent(ABC):
    """Abstract base class for a specialized agent."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a description of the agent."""
        pass

    def __init__(self, llm: BaseChatModel, name: Optional[str] = None):
        """Initialize the agent with a language model and optional name.
        
        Args:
            llm: The language model to use
            name: Optional custom name for the agent. If not provided, uses class name.
        """
        self.llm = llm
        self.name = name or self.__class__.__name__
        self.tool_logger = ToolCallLogger(self.name)
        self.agent = self._create_agent()

    @property
    @abstractmethod
    def tools(self) -> List[Callable]:
        """Return a list of tools for the agent."""
        pass

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Return the system prompt for the agent."""
        pass

    def _create_agent(self):
        """Create and compile the agent graph."""
        return create_agent(
            self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt
        )

    def invoke(self, *args, **kwargs):
        """Invoke the agent graph with tool call logging."""
        # Add the tool logger to callbacks if not already present
        if "config" not in kwargs:
            kwargs["config"] = {}
        if "callbacks" not in kwargs["config"]:
            kwargs["config"]["callbacks"] = []
        kwargs["config"]["callbacks"].append(self.tool_logger)
        return self.agent.invoke(*args, **kwargs)
