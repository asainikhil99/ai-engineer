import os
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

# ── LLM ──────────────────────────────────────────────
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

# ── Embeddings ────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# ── Load + Split PDF ──────────────────────────────────
def load_and_split(file_path):
    print("📄 Loading PDF...")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks")
    return chunks

# ── Vector Store ──────────────────────────────────────
def get_vectorstore(file_path=None):
    if os.path.exists("./chroma_langchain"):
        print("✅ Loading existing ChromaDB...")
        return Chroma(
            persist_directory="./chroma_langchain",
            embedding_function=embeddings
        )
    else:
        chunks = load_and_split(file_path)
        print("🔢 Embedding and storing chunks...")
        return Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./chroma_langchain"
        )

# ── RAG + Memory Chain ────────────────────────────────
def build_chain(vectorstore):
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a personal AI assistant for Sai Nikhil Avula,
a Senior AI/ML Engineer based in New York City.

Answer questions based ONLY on the context provided.
If the answer is not in the context say 'I don't have that information.'
Be friendly, concise and professional.

LINKS:
- Portfolio: https://asainikhil99.github.io/portfolioReact/
- GitHub: https://github.com/asainikhil99
- LeetCode: https://leetcode.com/u/saiNikhilAvula/

Context from resume:
{context}"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context": (lambda x: x["question"]) | retriever | format_docs,
            "question": lambda x: x["question"],
            "history": lambda x: x.get("history", [])
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever

# ── Main ──────────────────────────────────────────────
if __name__ == "__main__":
    print("🤖 Sai Nikhil's AI Assistant")
    print("-" * 40)

    if not os.path.exists("./chroma_langchain"):
        file_path = input("Enter resume PDF path: ")
        vectorstore = get_vectorstore(file_path)
    else:
        vectorstore = get_vectorstore()

    chain, retriever = build_chain(vectorstore)

    history = []

    print("\n💬 Assistant Ready! Type 'exit' to quit")
    print("-" * 40)

    while True:
        question = input("\nVisitor: ")
        if question.lower() == "exit":
            print("Goodbye! 👋")
            break

        answer = chain.invoke({
            "question": question,
            "history": history
        })

        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=answer))

        print(f"\n🤖 Assistant: {answer}")
        print("-" * 40)