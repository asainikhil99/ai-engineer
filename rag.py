import os
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="portfolio")

def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
    return chunks

def index_document(text, doc_id="resume"):
    print("📄 Chunking document...")
    chunks = chunk_text(text)
    print(f"✅ Created {len(chunks)} chunks")

    print("🔢 Embedding chunks...")
    for i, chunk in enumerate(chunks):
        embedding = embedding_model.encode(chunk).tolist()
        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[f"{doc_id}_chunk_{i}"]
        )
    print(f"✅ Stored {len(chunks)} chunks in ChromaDB")

def search(query, top_k=3):
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results['documents'][0]

def ask(question):
    chunks = search(question)
    context = "\n\n".join(chunks)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """
You are a strict Document Assistant. 
Use ONLY the provided context to answer the question. 
If the information is not in the context, respond with: 
"I am sorry, but that information is not available in the uploaded document."
Do not use your own outside knowledge.
"""
            },
            {
                "role": "user",
                "content": f"""Context:
                {context}
                
                Question: {question}"""
            }
        ]
    )
    return response.choices[0].message.content
print("_________RAG____________")
print("How do you want to give your input ?")
print("1. Paste/Type text")
print("2. Upload file")
inputType = input("Choose 1 or 2 : ")
if inputType == "1":
    text = input("Paste/Type your text : ")
elif inputType == "2":
    filePath = input("Enter file path : ")
    if filePath.endswith(".pdf"):
        import PyPDF2
        with open(filePath, "rb") as f:  # rb = read binary
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        print(f"✅ Loaded {len(text)} characters from PDF")
    else:
        with open(filePath, "r") as f:
            text = f.read()
        print(f"✅ Loaded {len(text)} characters")
print("🚀 Indexing resume...")
index_document(text)
print("\n💬 RAG System Ready!")
print("-" * 40)

while True:
    question = input("\nAsk anything about Sai: ")
    if question.lower() == "exit":
        break
    answer = ask(question)
    print(f"\n🤖 {answer}")
    print("-" * 40)