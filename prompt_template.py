import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

# intialize the connection to groq using API key
llm = ChatGroq(
    model = "llama-3.3-70b-versatile",
    api_key = os.getenv("GROQ_API_KEY")
)

# lets also keep track of history
converstionHistory = []

# PromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in {topic} be precise and clear and step by step ."),
    ("human", "{question}")
])

chain = prompt | llm

tests = [
    {"topic" : "AI Engineering", "question" : "What is RAG?"},
    {"topic" : "Cars", "question" : "Describe Porsche Gt3Rs and its pricing range."}
]


for test in tests:
    print("Topic : ", test["topic"])
    print("Question : ", test["question"])
    response = chain.invoke(test)
    print("Answer : ", response.content)
    print("-" * 40)



