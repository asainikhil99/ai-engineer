# Day 5: Semantic Meaning & Embeddings

## 🎯 Objective

To understand how AI "understands" relationships between words and sentences by converting text into numerical vectors.

## 🔄 The Logic Flow

1. **Input:** User provides a "Source Sentence" and a "Comparison Sentence."
2. **Vectorization:** Use `openai.embeddings.create` to get the numerical representation.
3. **Similarity Calculation:** Calculate the Cosine Similarity between the two vectors.
4. **Conclusion:** Determine if the sentences are semantically related based on a threshold (e.g., > 0.8).

## 🧪 Experiment Plan

- **Test 1:** Compare "I love coding in Python" vs "Programming in Python is great." (Expected: High Similarity)
- **Test 2:** Compare "I love coding in Python" vs "The moon is made of cheese." (Expected: Low Similarity)

## 💡 Key Takeaway

Embeddings allow us to build search engines that find _answers_ based on context, not just matching words.

### 🛡️ Troubleshooting: GCP Project Integration

Encountered "Unable to create API key" in AI Studio.

- **Resolution:** Manually created a Google Cloud Project and enabled the `Generative Language API`.
- **Learning:** API keys must be associated with a cloud project for quota and security management.
