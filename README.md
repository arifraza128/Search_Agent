# Web Research Agent (Search_Agent)

A lightweight, powerful LangGraph agent that automates web research. Given any query, the agent queries the web using Tavily Search, gathers information from the top 5 sources, and synthesizes it into a highly structured, objective research report using OpenAI's `gpt-4o-mini`.

## Architecture
The agent uses a two-node state machine graph powered by **LangGraph**:
```
User Query ──> [Search Node] ──> Tavily Web Search ──> [Synthesize Node] ──> GPT-4o-mini ──> Structured Report
```

## Features
- **LangGraph Orchestration**: Robust state management via `StateGraph`.
- **Tavily Web Search**: Fast, developer-friendly search retrieving 5 search results containing titles, URLs, and snippet contents.
- **GPT-4o-mini Synthesis**: Compiles raw findings into structured Markdown reports featuring:
  - **Summary**: High-level context of findings.
  - **Key Findings**: Detailed bullet points backed by sources.
  - **Sources**: Clear references to retrieved links.
- **Windows Console Resilient**: Reconfigures system encoding to UTF-8 dynamically to prevent Windows console encoding crashes when outputting emojis (`🔍`, `❌`).

## Setup

### Prerequisites
- Python 3.10+
- OpenAI API Key
- Tavily API Key

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/arifraza128/Search_Agent.git
   cd Search_Agent
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Setup environment variables:
   ```bash
   copy .env.example .env
   # Or on Mac/Linux: cp .env.example .env
   ```
   Open the `.env` file and insert your API keys:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   TAVILY_API_KEY=your_tavily_api_key
   ```

## Usage

Run the agent with the default research query (*"latest advances in AI agents 2024"*):
```bash
python agent.py
```

Or query a custom topic using the `--query` argument:
```bash
python agent.py --query "latest advances in quantum computing"
```

### Sample Output
```text
🔍 Researching: latest advances in AI agents 2024

============================================================
📄 RESEARCH REPORT
============================================================
## Summary
AI agents have seen significant advances in 2024, with major improvements
in reasoning, tool use, and multi-agent collaboration...

## Key Findings
- LangGraph and CrewAI have emerged as leading frameworks for production agents.
- OpenAI's GPT-4o enables real-time multimodal agent interactions.

## Sources
- https://...
```