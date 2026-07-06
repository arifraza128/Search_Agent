import os
import argparse
from typing import List
from typing_extensions import TypedDict
from dotenv import load_dotenv

# LangGraph imports
# pyrefly: ignore [missing-import]
from langgraph.graph import StateGraph, START, END

# Tavily & OpenAI imports
from tavily import TavilyClient
# pyrefly: ignore [missing-import]
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Define state structure
class AgentState(TypedDict):
    query: str
    search_results: List[dict]
    report: str

def search_node(state: AgentState):
    """
    Search node that queries Tavily for the search term.
    Retrieves the top 5 results and adds them to the state.
    """
    query = state["query"]
    print(f"🔍 Researching: {query}\n")
    
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        raise ValueError("TAVILY_API_KEY environment variable is not set. Please add it to your .env file.")
    
    # Initialize Tavily client
    client = TavilyClient(api_key=tavily_key)
    
    # Perform search (retrieve 5 results)
    response = client.search(query=query, max_results=5)
    results = response.get("results", [])
    
    cleaned_results = []
    for r in results:
        cleaned_results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", "")
        })
        
    return {"search_results": cleaned_results}

def synthesize_node(state: AgentState):
    """
    Synthesize node that compiles search results into a structured markdown report.
    Uses GPT-4o-mini.
    """
    query = state["query"]
    search_results = state["search_results"]
    
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set. Please add it to your .env file.")
        
    # Format the search results as clean text for the LLM
    formatted_results = ""
    for idx, res in enumerate(search_results, start=1):
        formatted_results += f"[{idx}] Title: {res['title']}\nURL: {res['url']}\nContent: {res['content']}\n\n"
        
    system_prompt = (
        "You are an expert research analyst. Synthesize findings from a list of web search results "
        "into a structured research report based on a user's query.\n\n"
        "The report MUST follow this exact markdown structure:\n"
        "## Summary\n"
        "[Provide a comprehensive, high-level summary of the research topic and what the findings indicate. Keep it professional, objective, and dense with information.]\n\n"
        "## Key Findings\n"
        "[List the key findings as bullet points. Each finding should be clear, detailed, and directly supported by the search results. Avoid generic bullet points.]\n\n"
        "## Sources\n"
        "[Provide a list of unique source URLs in bullet format, matching the layout requested by the user: - URL]"
    )
    
    user_prompt = (
        f"Research Query: {query}\n\n"
        f"Search Results:\n{formatted_results}\n\n"
        f"Please generate the structured report now."
    )
    
    # Initialize GPT-4o-mini
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=openai_key)
    
    # Run synthesis
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    
    return {"report": response.content}

# Build and compile graph
workflow = StateGraph(AgentState)
workflow.add_node("search", search_node)
workflow.add_node("synthesize", synthesize_node)

workflow.add_edge(START, "search")
workflow.add_edge("search", "synthesize")
workflow.add_edge("synthesize", END)

app = workflow.compile()

def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Web Research Agent using LangGraph, OpenAI, and Tavily.")
    parser.add_argument(
        "--query", 
        type=str, 
        default="latest advances in AI agents 2024",
        help="The research query to search and analyze."
    )
    args = parser.parse_args()
    
    openai_key = os.getenv("OPENAI_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")
    
    if not openai_key or not tavily_key:
        print("⚠️  Warning: API keys are missing from environment variables.")
        print("Please create a .env file containing OPENAI_API_KEY and TAVILY_API_KEY values.")
        print("You can copy the template using: Copy-Item .env.example .env (or cp .env.example .env)")
        return
        
    initial_state = {
        "query": args.query,
        "search_results": [],
        "report": ""
    }
    
    try:
        final_state = app.invoke(initial_state)
        
        print("============================================================")
        print("📄 RESEARCH REPORT")
        print("============================================================")
        print(final_state["report"])
    except Exception as e:
        print(f"❌ Error running agent: {e}")

if __name__ == "__main__":
    main()
