import os
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

# ── Tools ─────────────────────────────────────────────
@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression.
    Input should be a valid Python math expression like '2 + 2' or '15 * 0.20'"""
    try:
        result = eval(expression)
        return str(result)
    except:
        return "Invalid expression"

@tool
def get_resume_info(query: str) -> str:
    """Get information about Sai Nikhil's experience,
    skills, education and background.
    Use this for any questions about Sai."""
    info = {
        "experience": "5.5 years in ML engineering at Meta, J&J, Goldman Sachs",
        "skills": "Python, RAG, LLMs, LangChain, FastAPI, Docker, AWS",
        "education": "MS Computer Science, Stevens Institute of Technology 2022-2024",
        "meta": "Senior AI/ML Engineer, built RAG system improving relevance by 30%",
        "goldman": "ML Engineer, built fraud detection saving $55K-$60K annually",
        "links": "GitHub: github.com/asainikhil99, Portfolio: asainikhil99.github.io/portfolioReact"
    }

    query_lower = query.lower()
    results = []

    for key, value in info.items():
        if key in query_lower or any(word in query_lower for word in value.lower().split()):
            results.append(value)

    return "\n".join(results) if results else "\n".join(info.values())

tools = [calculate, get_resume_info]

# ── Create Agent ──────────────────────────────────────
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful AI assistant for Sai Nikhil Avula. Always use tools when you need information rather than guessing. Think step by step."
)

# ── Main ──────────────────────────────────────────────
if __name__ == "__main__":
    print("🤖 Sai's AI Agent")
    print("-" * 40)
    print("Tools: calculator, resume info")
    print("-" * 40)

    while True:
        question = input("\nYou: ")
        if question.lower() == "exit":
            print("Goodbye!")
            break

        result = agent.invoke({"messages": [{"role": "user", "content": question}]})
        answer = result["messages"][-1].content
        print(f"\n🤖 Agent: {answer}")
        print("-" * 40)