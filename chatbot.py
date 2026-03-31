import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

conversation_history = []

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
    print(conversation_history)
    print(f"AI: {assistant_reply}")
    print("-" * 40)
