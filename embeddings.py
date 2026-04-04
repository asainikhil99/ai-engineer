# import numpy as np
# from sentence_transformers import SentenceTransformer
# import time

# # 1. Load the model 
# # The first time you run this, it will download (~80MB). 
# # After that, it runs 100% offline from your hard drive.
# print("⏳ Loading local embedding model...")
# model = SentenceTransformer('all-MiniLM-L6-v2') 

# def get_embedding(text):
#     """
#     Directly converts text to a 384-dimension vector on your CPU.
#     No API keys, no 404 errors, no cost.
#     """
#     return model.encode(text)

# def calculate_similarity(v1, v2):
#     """
#     Your DSA heart: Cosine Similarity 
#     Formula: (A . B) / (||A|| * ||B||)
#     """
#     dot_product = np.dot(v1, v2)
#     norm_a = np.linalg.norm(v1)
#     norm_b = np.linalg.norm(v2)
#     return dot_product / (norm_a * norm_b)

# # --- THE TEST SUITE ---
# if __name__ == "__main__":
#     print("🚀 Local Semantic Search initialized.")

#     # Define our test cases
#     source = "I am a software engineer who loves Python."
    
#     # Case 1: High Similarity (Related context)
#     match = "Coding in Python is my professional focus."
    
#     # Case 2: Low Similarity (Different context)
#     random = "I prefer eating pizza with extra cheese."

#     print(f"\nProcessing: '{source}'")
    
#     # Generate Vectors
#     start = time.time()
#     vec_source = get_embedding(source)
#     vec_match = get_embedding(match)
#     vec_random = get_embedding(random)
#     end = time.time()

#     # Calculate Scores
#     score_1 = calculate_similarity(vec_source, vec_match)
#     score_2 = calculate_similarity(vec_source, vec_random)

#     print("\n" + "="*45)
#     print(f"Match Score:  {score_1:.4f} ✅")
#     print(f"Random Score: {score_2:.4f} ❌")
#     print(f"Compute Time: {end - start:.4f} seconds")
#     print("="*45)
    
#     # Engineering Check: Print the vector shape
#     print(f"\nVector Dimensions: {len(vec_source)}") 
#     print(f"First 5 numbers: {vec_source}")


import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "I love my puppy",
    "I like dog",
    "Python is a great programming language",
    "I enjoy coding in Python",
    "The stock market crashed today",
    "Quantum physics is fascinating",
    "AI is transforming software engineering",
    "Machine learning models need lots of data"
]

query = "I like dogs"

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(f"🔍 Query: '{query}'")
print("-" * 40)

query_embedding = model.encode(query) # black box (tokenize + transform + pool)

results = []
for sentence in sentences:
    embedding = model.encode(sentence) # black box (tokenize + transform + pool)
    similarity = cosine_similarity(query_embedding, embedding)
    results.append((sentence, similarity))

results.sort(key=lambda x: x[1], reverse=True)

print("📊 Most similar sentences:")
for i, (sentence, score) in enumerate(results):
    print(f"{i+1}. {sentence}")
    print(f"   Similarity: {score:.4f}")
    print()