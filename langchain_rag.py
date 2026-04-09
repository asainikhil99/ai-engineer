import os
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
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

    print("✂️  Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks")
    return chunks

# ── Index into ChromaDB ───────────────────────────────
def create_vectorstore(chunks):
    print("🔢 Embedding and storing chunks...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_langchain"
    )
    print("✅ Stored in ChromaDB")
    return vectorstore

# ── Load existing ChromaDB ────────────────────────────
def load_vectorstore():
    print("✅ Loading existing ChromaDB...")
    return Chroma(
        persist_directory="./chroma_langchain",
        embedding_function=embeddings
    )

# ── Build RAG Chain ───────────────────────────────────
def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are Sai Nikhil's personal assistant.
         Answer questions based ONLY on the following context.
         If the answer is not in the context say
         'I don't have that information.'

         Context: {context}"""),
        ("human", "{question}")
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever

# ── Main ──────────────────────────────────────────────
if __name__ == "__main__":
    print("🤖 LangChain RAG System")
    print("-" * 40)

    if os.path.exists("./chroma_langchain"):
        vectorstore = load_vectorstore()
    else:
        file_path = input("Enter PDF path: ")
        chunks = load_and_split(file_path)
        vectorstore = create_vectorstore(chunks)

    chain, retriever = build_rag_chain(vectorstore)

    print("\n💬 RAG Ready! Type 'exit' to quit")
    print("-" * 40)

    while True:
        question = input("\nAsk: ")
        if question.lower() == "exit":
            break

        answer = chain.invoke(question)
        print(f"\n🤖 {answer}")

        docs = retriever.invoke(question)
        print("\n📚 Sources used:")
        for i, doc in enumerate(docs):
            print(f"  Chunk {i+1}: {doc.page_content[:100]}...")
        print("-" * 40)