1. What is Pre-training and Fine tuning ?
   1.1 Pre-training is done once in a while where the big tech companies take chunk of data from internet and train them.
   1.2 Fine tuning is taking the already trained model and training it further on specific data to make it
   better at a specific task.
   Example: base model just completes text → fine tune it with human feedback (RLHF) → becomes a helpful assistant like ChatGPT.

2. What are parameters — what is your best analogy for it?
   Parameters are the learnable weights of a neural network adjusted during training via backpropagation.

   Same 2 trillion tokens of experience
   ↓
   7B = like a person with average memory
   remembers the big patterns

   70B = like a person with photographic memory
   remembers big patterns AND subtle nuances.

3. What is the difference between training data and parameters?

   training data - has all the repeated words like actual text, data.
   parameters - has built numbers based on the words.

4. What are scaling laws — why do bigger models get smarter?
   -- Scaling laws mean that if you give an LLM more parameters, data, or compute it gets better in a predictable way.

5. What is RAG and why does it exist?
   RAG is a memory for LLM

6. What is the reversal curse — use your own example?
   Reversal Curse means the knowledge to the model is trained in one direction.

7. What is the context window?
   Context window is everything the model can see in a single converation. It includes system prompt, user messages, assistant history, and RAG injected facts. When the conversation ends it gets deleted."

8. What is prompt injection and why should I care as a developer?
   "Prompt injection is when a malicious user tries to manipulate your model through their prompt message at runtime."

9. One thing that surprised me today?
   I never knew that models are trained once in while with huge chunks of data from internet and its done in one direction

10. One thing I still find confusing?
    if the model is trained and the intution it built is called parameters how they are stored and how it will answer based in it

--------------------------------------------- DAY 4 --------------------------------------------------------

1. Knowledge gets cutoff after training.
   Example : ask chatbot about present updates it can only give updates until dec 2023 because that is the date it is trained on.
   To get latest updates we use RAG.
