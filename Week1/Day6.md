Day 6: Vector Databases & Document Strategy
🏗️ 1. Chunking Strategies: The Art of Splitting
Before we turn text into vectors, we have to decide how to "slice" the data. If a chunk is too small, it loses meaning. If it's too big, the embedding becomes "blurry."

Strategy Logic Pros/Cons
Fixed Size Split every N characters/words. Simple, but breaks sentences mid-thought. ❌
Recursive Split by Paragraph → Sentence → Word. Standard. Preserves hierarchy. (LangChain default) ✅
Sliding Window N words with X word overlap. Context is preserved across chunks. ✅
Semantic Split only when the "meaning" changes. Elite. Uses AI to find topic boundaries. 🏆
Document Specific Split by Markdown headers or code functions. Best for technical docs and GitHub repos. ✅

🧠 2. Embedding Models: Choosing the "Lens"
The model you choose defines the "resolution" of your semantic map.

all-MiniLM-L6-v2 (Local): 384 dimensions. Ultra-fast, perfect for local development and CPU-only environments. (Our choice for Day 5/6).

all-mpnet-base-v2 (Local): 768 dimensions. Slower than MiniLM but significantly more accurate for complex nuances.

BGE-M3 (Open Source): The current "King" of open-source embeddings. Handles multi-lingual data brilliantly.

OpenAI text-embedding-3-small (Cloud): 1536 dimensions. High accuracy, but requires an API key and internet.

Cohere Embed (Cloud): Specifically optimized for "Retrieval" tasks and massive datasets.

🏛️ 3. Vector Databases: The Long-Term Memory
Traditional databases (SQL) index by keywords; Vector DBs index by distance.

Database Best For... Key Feature
ChromaDB Learning & Local Apps Zero-config, runs in your project folder. ✅
Pinecone Production Scale Cloud-native, "managed" so you don't handle servers.
FAISS (Meta) Raw Speed Extremely fast similarity search, but no built-in storage.
pgvector Existing Apps An extension for PostgreSQL. Use it if you already have a SQL DB.
Weaviate / Qdrant Enterprise Hybrid Great for combining Vector search with Keyword search.

To be a "Good AI Engineer," I need to understand that Retrieval is 80% of the battle. \* If my Chunking is bad, the AI gets the wrong context.

If my Embedding is weak, the AI can't find the right "neighborhood."

If my Vector DB isn't optimized, the app will be too slow for users.
