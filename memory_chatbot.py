import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv

load_dotenv()

# intialize the llm
llm = ChatGroq(
    api_key = os.getenv("GROQ_API_KEY"),
    model = "llama-3.3-70b-versatile"
)

# lets also keep track of history
history = []

# prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a personal assistant for Sai Nikhil,
     an AI engineer based in New York City.
     Be friendly, concise and professional.
     RULES:
     - Never make up anything if you dont know something say 'I am not aware of it'"""),
     MessagesPlaceholder(variable_name= "history"),
("human", "{question}")
])

chain = prompt | llm
question = input("You : ")
while question.lower() != "exit":
    response = chain.invoke({
        "history" : history, 
        "question" : question
    })
    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=response.content))
    print(response.content)
    print("-" * 40)
    question = input("You : ")

print("Goodbye")



