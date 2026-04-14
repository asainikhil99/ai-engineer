import os
from typing import TypedDict, Annotated
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

# ── State ─────────────────────────────────────────────
class State(TypedDict):
    messages: Annotated[list, add_messages]

# ── Tools ─────────────────────────────────────────────
@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression like '2 + 2' or '15 * 0.20'"""
    try:
        return str(eval(expression))
    except:
        return "Invalid expression"

@tool
def get_resume_info(query: str) -> str:
    """Get info about Sai Nikhil's experience, skills, and background."""
    info = [
        "5.5 years ML engineering at Meta, J&J, Goldman Sachs",
        "Skills: Python, RAG, LLMs, LangChain, FastAPI, Docker, AWS",
        "MS Computer Science, Stevens Institute of Technology 2022-2024",
        "At Meta: built RAG system improving relevance by 30%",
        "At Goldman: fraud detection saving $55K-$60K annually",
    ]
    return "\n".join(info)

tools = [calculate, get_resume_info]
llm_with_tools = llm.bind_tools(tools)

# ── Nodes ─────────────────────────────────────────────
def call_llm(state: State) -> State:
    """Node 1: call the LLM with current messages"""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def run_tools(state: State) -> State:
    """Node 2: execute whatever tool the LLM asked for"""
    last_message = state["messages"][-1]
    tool_map = {t.name: t for t in tools}
    results = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        result = tool_map[tool_name].invoke(tool_args)

        from langchain_core.messages import ToolMessage
        results.append(ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"]
        ))

    return {"messages": results}

# ── Router ────────────────────────────────────────────
def should_use_tool(state: State) -> str:
    """Decide: did the LLM want a tool, or is it done?"""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "run_tools"
    return END

# ── Build Graph ───────────────────────────────────────
graph = StateGraph(State)

graph.add_node("call_llm", call_llm)
graph.add_node("run_tools", run_tools)

graph.add_edge(START, "call_llm")
graph.add_conditional_edges("call_llm", should_use_tool)
graph.add_edge("run_tools", "call_llm")  # loop back after tool

app = graph.compile()

# ── Main ──────────────────────────────────────────────
if __name__ == "__main__":
    print("🤖 LangGraph Agent")
    print("-" * 40)

    messages = [
        SystemMessage(content="You are a helpful assistant for Sai Nikhil. Use tools when needed.")
    ]

    while True:
        question = input("\nYou: ")
        if question.lower() == "exit":
            break

        messages.append(HumanMessage(content=question))
        result = app.invoke({"messages": messages})
        messages = result["messages"]

        answer = messages[-1].content
        print(f"\n🤖 Agent: {answer}")
        print("-" * 40)