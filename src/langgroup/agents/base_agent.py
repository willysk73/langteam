
"""Base class for creating specialized agents."""
from abc import ABC, abstractmethod
from typing import List, Callable, Optional
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel


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
        """Invoke the agent graph."""
        return self.agent.invoke(*args, **kwargs)
