# LangTeam

A multiagent system framework built on LangChain and LangGraph with supervisor-based coordination.

LangTeam provides a flexible architecture for creating teams of specialized AI agents that collaborate on complex tasks under the guidance of a supervisor agent.

## Features

- **Supervisor Architecture**: Intelligent task routing and coordination
- **Extensible Agent System**: Easy-to-extend base classes for custom agents
- **LangGraph Integration**: Stateful workflows with LangGraph
- **Type-Safe**: Full type hints and Pydantic models

## Installation

```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Quick Start

1. Set up your environment:
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

2. Create your custom agents by extending `BaseAgent`:
```python
from langteam import BaseAgent
from typing import List, Callable

class MyAgent(BaseAgent):
    @property
    def description(self) -> str:
        return "Description of what this agent does"
    
    @property
    def tools(self) -> List[Callable]:
        return [my_tool_function]
    
    @property
    def system_prompt(self) -> str:
        return "System prompt for the agent"
```

3. Set up the agent system:
```python
from langteam import AgentSystem
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agents = [MyAgent(llm), AnotherAgent(llm)]
system = AgentSystem(llm, agents)

result = system.run("Your task here")
```

## Usage

### Single Agent (Basic)
Run the basic agent:
```bash
python main.py
```

The agent includes two tools:
- **Calculator**: Performs mathematical calculations
- **Weather**: Returns weather information (mock implementation)

### Multiagent System with Supervisor
Run the multiagent system:
```bash
python multiagent_supervisor.py
```

The multiagent system includes:
- **Supervisor Agent**: Orchestrates and routes tasks to specialized agents
- **Research Agent**: Handles research and information gathering
- **Analysis Agent**: Analyzes data and provides insights
- **Writing Agent**: Writes and formats content
- **Math Agent**: Performs calculations

The supervisor dynamically decides which agents to invoke and coordinates their work, allowing agents to collaborate back and forth on complex tasks.

## Architecture

The multiagent system uses LangGraph to create a stateful workflow where:
1. Supervisor receives the task
2. Supervisor routes to appropriate sub-agent
3. Sub-agent completes its work
4. Control returns to supervisor
5. Process repeats until task is complete

## Customization

Add your own tools by creating functions and registering them with the `Tool` class in either `main.py` or `multiagent_supervisor.py`.
