
# RAG System Experiments

RAG(Retrieval-Augmented Generation) 시스템의 검색 성능 최적화를 위한 실험 프로젝트입니다.

FAISS 기반 Vector Retrieval의 검색 정확도와 Latency를 측정하고, 검색 파라미터 및 Corpus 규모에 따른 성능 변화를 분석합니다.

## Models & Components

- **Embedding:** BAAI/bge-m3 (1024 dim, L2 normalized)
- **Vector Search:** FAISS (IndexFlatIP / IndexIVFFlat)
- **Generation:** OpenAI GPT-OSS 20B via Groq (Initial RAG Pipeline)

## Experiments

### 1. Initial RAG Pipeline
소규모 문서 청크를 대상으로 RAG Pipeline을 구현하고 단계별 Latency를 측정했습니다.

### 2. SciFact Retrieval Experiment
BEIR SciFact 데이터셋(5,183 documents)을 활용하여 `nlist` 및 `nprobe` 변화에 따른 검색 성능을 분석했습니다.

### 3. NQ Corpus Scale Experiment
BEIR NQ 데이터셋을 활용하여 Corpus 규모(10K / 30K / 50K)별 Retrieval Quality–Latency Trade-off를 분석했습니다.

[View NQ Experiment Results](experiments/nq/README.md)
