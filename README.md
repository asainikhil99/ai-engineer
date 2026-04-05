# AI Engineer Learning Journey 🤖

A collection of AI engineering projects built during Week 1
of my AI engineer roadmap.

## Projects

### 1. First API Call (`firstAPICall.py`)

My first ever LLM API call using Groq + Llama 3.3

### 2. CLI Chatbot (`chatbot.py`)

A conversational chatbot with:

- Custom personality via system prompt
- Conversation memory via context window
- Prompt injection defense
- Portfolio links

### 3. Document Summarizer (`summarizer.py`)

Summarizes any text or file in 4 styles:

- Bullet points
- ELI5 (Explain like I'm 5)
- Executive brief
- Key takeaways

### 4. Embeddings + Semantic Search (`embeddings.py`)

Semantic search using sentence transformers:

- Text → embeddings via all-MiniLM-L6-v2
- Cosine similarity for ranking
- Finds meaning not just keywords

### 5. RAG Pipeline (`rag.py`)

Full Retrieval Augmented Generation pipeline:

- PDF parsing
- Text chunking with overlap
- ChromaDB vector storage
- Semantic search + context injection

### 6. Portfolio Chatbot (`portfolio_chatbot.py`)

Final combined project — RAG + Chatbot:

- Reads resume PDF
- Answers questions about Sai Nikhil
- Never hallucinates — only answers from resume
- Remembers conversation history
- Prompt injection defense

## Tech Stack

- Python 3.13
- Groq API (Llama 3.3 70B)
- Sentence Transformers
- ChromaDB
- PyPDF2

## Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install groq openai python-dotenv sentence-transformers chromadb PyPDF2 numpy

# Add your API key
echo "GROQ_API_KEY=your_key_here" > .env

# Run portfolio chatbot
python3 portfolio_chatbot.py
```

## What I Learned

- How LLMs work (tokens, parameters, context window)
- Prompt engineering (system prompts, few shot, security)
- Embeddings and semantic search
- RAG pipeline end to end
- Git workflow for AI projects
