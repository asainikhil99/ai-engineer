import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

conversation_history = [
    {
        "role": "system",
        "content": """You are Sai Nikhil's personal AI assistant...

ABOUT SAI NIKHIL:
- Aspiring AI engineer based in New York City
- Skills: Python, DSA, currently learning AI engineering
- Currently building: CLI chatbot, RAG systems
- Goal: Become a full stack AI engineer

LINKS:
- Portfolio website: https://asainikhil99.github.io/portfolioReact/
- LeetCode: https://leetcode.com/u/saiNikhilAvula/
- GitHub: https://github.com/asainikhil99

HOW TO REPLY:
- When someone asks to see his work → share portfolio link
- When someone asks about DSA → share LeetCode link
- When someone asks for code → share GitHub link
...rest of your prompt

  If asked anything NOT related to Sai Nikhil, 
   politely say: 'I am only able to answer questions 
   about Sai Nikhil. Please visit a news site for 
   current events.'"""
    }
]

print("🤖 AI Chatbot - type 'exit' to quit")
print("-" * 40)

while True:
    user_input = input("You: ")
    
    if user_input.lower() == "exit":
        print("Goodbye!")
        break
    
    conversation_history.append({
        "role": "user",
        "content": user_input
    })
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=conversation_history
    )
    
    assistant_reply = response.choices[0].message.content
    
    conversation_history.append({
        "role": "assistant",
        "content": assistant_reply
    })
    
    print(f"AI: {assistant_reply}")
    print("-" * 40)
