from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

# History starts with system prompt
history = [
    SystemMessage(content="You are a helpful assistant.")
]

print("💬 Chatbot with memory — type 'exit' to quit")
print("-" * 40)

while True:
    text = input("\nYou: ")
    if text.lower() == "exit":
        print("Goodbye!")
        break

    # Add user message to history
    history.append(HumanMessage(content=text))

    # Send full history to LLM
    response = llm.invoke(history)

    # Add AI reply to history
    history.append(AIMessage(content=response.content))

    print(f"AI: {response.content}")

    print("-" * 40)
    print(history)