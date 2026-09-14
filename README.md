# RAG System Experiments

## Models & Components

This project currently uses the following models and components for the baseline RAG pipeline.

### Embedding Model
- **Model:** BAAI/bge-m3
- **Library:** Sentence Transformers
- **Embedding Dimension:** 1024
- **Normalization:** L2 normalization enabled

### Vector Search
- **Library:** FAISS (Facebook AI Similarity Search)
- **Index:** IndexFlatIP
- **Search Type:** Exact Search
- **Similarity:** Inner Product on normalized embeddings
  - Equivalent to Cosine Similarity when vectors are L2-normalized

### Generation Model
- **Model:** OpenAI GPT-OSS 20B
- **Model ID:** `openai/gpt-oss-20b`
- **API Provider:** Groq
- **API Interface:** OpenAI-compatible API

### Retrieval Configuration
- **Chunking:** Paragraph-based chunking
- **Default Top-K:** 3
- **Reranking:** Not applied in the current baseline