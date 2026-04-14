# Day 1 Interview Prep — RAG

## Q1: What is RAG, why does it exist, and what problem does it solve?

RAG stands for Retrieval Augmented Generation. LLMs have two core problems —
their knowledge is frozen at training time, and they hallucinate because they
generate based on patterns not facts. There's also the reversal curse — if the
model was trained on "Deepika's husband is Nikhil" it can't reverse that to
answer "who is Nikhil's wife." RAG solves all of this by retrieving relevant
documents from a vector database at query time, injecting them into the prompt
as context, and then letting the model generate a grounded response. At Meta I
used this for the AI Assistant so responses were based on real retrieved data
rather than what the model memorized.

**Key insight for interviews:**
"RAG doesn't just reduce hallucinations — it fundamentally changes the task
from recall to comprehension. LLMs are terrible at recall, great at
comprehension."

## Q2: Walk me through the RAG pipeline end to end.

**Indexing phase (happens once, offline):**

1. Take your document
2. Chunk it into pieces (typically 256-512 tokens with 10-20% overlap)
3. Embed each chunk using an embedding model (sentence-transformers)
4. Store the vectors + original text in a vector DB (ChromaDB, FAISS, Pinecone)

**Query phase (happens every time a user asks something):**

1. Embed the user question
2. Search vector DB for most similar chunks (cosine similarity)
3. Retrieve top K chunks
4. Inject chunks into the prompt as context
5. LLM reads the context and generates a grounded answer

## Q3: How did you measure the 30% improvement in response relevance?

We measured using a combination of offline and online evaluation:

**Retrieval metrics:**

- Recall@K — out of all relevant chunks that exist, how many did we find in
  top K results?
- MRR (Mean Reciprocal Rank) — how high up in ranked results does the first
  relevant chunk appear? Score of 1.0 means it's always first.

**RAG-specific metrics:**

- Groundedness — how well the response is supported by retrieved source
  documents. Measures whether every claim in the answer can be traced back
  to a retrieved chunk.
- Hallucination rate — percentage of responses containing unsupported claims.

**Online signals:**

- A/B test between baseline and improved pipeline
- Human annotators scored responses 1-5 on relevance, correctness,
  completeness
- Reduced query reformulation rate (users getting answers in fewer attempts)

## Q4: What are the tradeoffs of chunk size?

**Too small:**
Loses context. A chunk like "He joined in 2019" means nothing without
surrounding sentences. Embedding captures an incomplete thought, retrieval
quality drops.

**Too large:**
Introduces noise. If a chunk contains 5 different topics, its embedding
becomes an average of all of them — doesn't represent any single topic well.
Might retrieve a chunk that's only 10% relevant.

**Sweet spot:**
256-512 tokens with 10-20% overlap. Overlap ensures sentences on chunk
boundaries don't lose context.

**One-liner:**
"Chunk size is a retrieval quality tradeoff — too small loses context, too
large introduces noise. We tuned ours to 512 tokens with 20% overlap based
on evaluation metrics."

## Key Terms to Know

| Term              | Definition                                                                 |
| ----------------- | -------------------------------------------------------------------------- |
| RAG               | Retrieval Augmented Generation — grounding LLM responses in retrieved docs |
| Reversal Curse    | LLMs can't reverse relationships they were trained on directionally        |
| Chunking          | Splitting documents into smaller pieces for embedding                      |
| Embedding         | Converting text to a vector of numbers representing meaning                |
| Cosine Similarity | Measures angle between two vectors — closer = more similar meaning         |
| Vector DB         | Database that stores and searches embeddings (ChromaDB, FAISS, Pinecone)   |
| Recall@K          | Of all relevant chunks, how many appear in top K results                   |
| MRR               | Mean Reciprocal Rank — how high up the first relevant chunk appears        |
| Groundedness      | How well LLM response is supported by retrieved source documents           |
| Hallucination     | LLM generating confident but unsupported/false information                 |

## Day 1 Score

| Topic                 | Score |
| --------------------- | ----- |
| RAG definition        | 9/10  |
| Why RAG exists        | 9/10  |
| Indexing + query flow | 10/10 |
| Evaluation metrics    | 8/10  |
| Chunking tradeoffs    | 7/10  |

**Tomorrow — Day 2: Hybrid RAG (dense + sparse retrieval)**
