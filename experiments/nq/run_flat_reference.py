import time

import faiss
import numpy as np


########## 설정 ##########

DOCUMENT_EMBEDDING_PATH = "data/nq_corpus_500k_embeddings.npy"
QUERY_EMBEDDING_PATH = "data/nq_test_query_embeddings.npy"

TOP_K = 10

CORPUS_SIZES = [
    100_000,
    300_000,
    500_000
]


########## Embedding 로드 ##########

document_embeddings = np.load(
    DOCUMENT_EMBEDDING_PATH,
    mmap_mode="r"
)

query_embeddings = np.load(
    QUERY_EMBEDDING_PATH
).astype(np.float32)

dimension = document_embeddings.shape[1]

print("===== FlatIP Reference =====")
print("Document Embeddings:", document_embeddings.shape)
print("Query Embeddings:", query_embeddings.shape)
print("Dimension:", dimension)


########## Corpus Size별 FlatIP 검색 ##########

for corpus_size in CORPUS_SIZES:

    print("")
    print("=" * 60)
    print(f"Corpus Size: {corpus_size:,}")
    print("=" * 60)

    corpus_vectors = np.asarray(
        document_embeddings[:corpus_size],
        dtype=np.float32
    )

    ########## Index 생성 ##########

    index = faiss.IndexFlatIP(dimension)

    index.add(corpus_vectors)

    print("Index Vectors:", index.ntotal)


    ########## Search ##########

    start = time.perf_counter()

    scores, indices = index.search(
        query_embeddings,
        TOP_K
    )

    end = time.perf_counter()

    total_latency = end - start

    average_latency_ms = (
        total_latency
        / len(query_embeddings)
        * 1000
    )


    ########## 결과 출력 ##########

    print(
        "Total Search Time:",
        round(total_latency, 4),
        "seconds"
    )

    print(
        "Average Latency:",
        round(average_latency_ms, 4),
        "ms/query"
    )

    print(
        "Result Shape:",
        indices.shape
    )


    ########## 결과 저장 ##########

    np.save(
        f"data/nq_flat_{corpus_size}_top10_indices.npy",
        indices
    )

    np.save(
        f"data/nq_flat_{corpus_size}_top10_scores.npy",
        scores
    )

    print("Reference Saved")