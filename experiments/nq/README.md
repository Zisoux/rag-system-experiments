# [ RAG System Experiments ]


## ✅ Models & Components

- **Embedding:** BAAI/bge-m3 (1024 dim, L2 normalized)
- **Vector Search:** FAISS (IndexFlatIP / IndexIVFFlat)


## NQ Corpus Scale Experiment

SciFact 실험에서는 Corpus 규모가 작아 높은 Recall을 확보할 경우 IVFFlat의 Latency 이점이 크지 않았습니다.

이에 따라 BEIR NQ 데이터셋을 활용하여 Corpus 규모별 `nprobe` 변화에 따른 Retrieval Quality–Latency Trade-off를 분석했습니다.

### Experiment Setup

| Item | Setting |
|---|---|
| Dataset | BEIR NQ |
| Corpus Size | 10K / 30K / 50K |
| Test Queries | 3,452 |
| Relevant Documents | 4,201 |
| Embedding | BAAI/bge-m3 (1024 dim, L2 normalized) |
| Baseline | FAISS IndexFlatIP |
| Comparison | FAISS IndexIVFFlat |
| nlist | 128 / 256 / 512 |
| nprobe | 1, 2, 4, 8, 16, 32, 64, 128 |
| Top-K | 10 |

모든 Relevant Document를 포함하고 Random Distractor를 추가하여 `10K ⊂ 30K ⊂ 50K` 형태로 Corpus를 구성했습니다.

### Results

| Corpus | FlatIP Latency | IVF nprobe=16 Recall@10 | IVF nprobe=16 Latency |
|---|---:|---:|---:|
| 10K | 0.0451 ms | 0.9387 | 0.0398 ms |
| 30K | 0.1524 ms | 0.9097 | 0.1316 ms |
| 50K | 0.3086 ms | 0.8791 | 0.1438 ms |

### nprobe vs Recall@10

![NQ nprobe vs Recall](experiments/nq/figures/nprobe_recall.png)

### nprobe vs Retrieval Latency

![NQ nprobe vs Latency](experiments/nq/figures/nprobe_latency.png)

### Quality-Latency Trade-off

![NQ Quality-Latency Trade-off](experiments/nq/figures/accuracy_latency_tradeoff.png)


### Analysis

- `nprobe` 증가에 따라 Recall@10과 FlatIP Top-10 Overlap이 향상되었습니다.
- 탐색 Vector 수와 Retrieval Latency도 함께 증가하여 Quality–Latency Trade-off를 확인했습니다.
- 높은 `nprobe` 구간에서는 검색 비용 증가 대비 Recall 개선 폭이 감소했습니다.
- 동일한 `nprobe`에서 Corpus 규모가 커질수록 Recall이 낮아지는 경향을 보였습니다. 단, Corpus별 `nlist`도 다르게 설정했습니다.
- 10K에서 `nprobe=128`은 FlatIP와 동일한 Top-10을 반환했지만, Latency는 더 높았습니다.

전체 결과는 `data/nq_ivf_small_results.csv`에 저장했습니다.

※ Latency는 단일 실행 기준이며, 반복 측정을 통한 검증은 추후 진행할 예정입니다.
