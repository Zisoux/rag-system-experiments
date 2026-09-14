# RAG System Experiments


## Models & Components

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


## Baseline Performance

현재 구현한 기본 RAG 파이프라인의 단계별 Latency를 확인하기 위해 간단한 성능 측정을 진행했습니다.

- **Query:** `Astra는 해킹 능력이 어느 정도야?`
- **Top-K:** 3
- **Runs:** 10
- **Dataset:** 9개 청크(문단 단위로 분할)

| Stage | Average | Minimum | Maximum |
|---|---:|---:|---:|
| Query Embedding | 158.33 ms | 116.70 ms | 275.32 ms |
| FAISS Retrieval | 0.06 ms | 0.04 ms | 0.15 ms |
| Generation | 721.66 ms | 441.46 ms | 1628.31 ms |
| End-to-End | 880.07 ms | 568.80 ms | 1903.75 ms |

현재의 소규모 Baseline 환경에서는 Generation 단계가 전체 End-to-End Latency의 대부분을 차지하는 것을 확인했습니다.

FAISS Retrieval의 경우 현재 Dataset이 9개의 Vector로 구성되어 있어 검색 시간이 매우 짧게 측정되었습니다. 따라서 현재 결과만으로 Retrieval 성능에 대한 일반적인 결론을 내리기에는 Dataset 규모가 작으며, 추후 데이터 규모를 확장한 실험이 필요합니다.

> **참고:** 현재 측정한 Generation Latency에는 외부 LLM API 요청 및 응답 시간이 포함되어 있으므로, 순수한 모델 Inference Time을 의미하지 않습니다.