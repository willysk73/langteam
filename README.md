# LangChain Agents Project

A project demonstrating LangChain agents, including a multiagent system with supervisor coordination.

## Setup

1. Install dependencies:
```bash
pip install -e .
```

2. Create a `.env` file with your OpenAI API key:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
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
