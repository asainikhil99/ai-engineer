import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def summarize(text, style):
    prompts = {
        "1": f"""Summarize the following text in clear bullet points.
                Each bullet should be one key idea.
                Max 5 bullet points.
                
                Text: {text}""",
                
        "2": f"""Explain the following text like I am 5 years old.
                Use simple words, short sentences, and a fun analogy.
                Max 3 sentences.
                
                Text: {text}""",
                
        "3": f"""Write an executive brief of the following text.
                Format: 
                - One line summary
                - 3 key points
                - One action item
                
                Text: {text}""",
                
        "4": f"""Extract the key takeaways from the following text.
                Format as numbered list.
                Max 4 takeaways.
                
                Text: {text}"""
    }
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompts[style]}
        ]
    )
    
    return response.choices[0].message.content

print("📄 Document Summarizer")
print("-" * 40)

text = input("Paste your text here:\n")

print("\nChoose summary style:")
print("1. Bullet points")
print("2. ELI5 (Explain like I'm 5)")
print("3. Executive brief")
print("4. Key takeaways")

style = input("\nEnter number (1-4): ")

print("\n📝 Summary:")
print("-" * 40)
print(summarize(text, style))