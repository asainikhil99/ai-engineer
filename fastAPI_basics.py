from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph_basics import app as agent

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def home():
    return {"status": "Nikhil's portfolio API is live"}

@app.post("/chat")
async def chat(request: ChatRequest):
    messages = [
        SystemMessage(content="You are a helpful assistant for Sai Nikhil. Only answer questions about his background, skills, and experience."),
        HumanMessage(content=request.message)
    ]
    result = agent.invoke({"messages": messages})
    return {"response": result["messages"][-1].content}