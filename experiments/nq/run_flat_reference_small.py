import time
from pathlib import Path

import faiss
import numpy as np

DOCUMENT_PATH = "data/nq_corpus_500k_embeddings.npy"
QUERY_PATH = "data/nq_test_query_embeddings.npy"

CORPUS_SIZES = [10_000, 30_000, 50_000]
TOP_K = 10

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

document_embeddings = np.load(
    DOCUMENT_PATH,
    mmap_mode="r"
)

query_embeddings = np.ascontiguousarray(
    np.load(QUERY_PATH),
    dtype=np.float32
)

dimension = document_embeddings.shape[1]

print("===== Small Corpus FlatIP Reference =====")

for corpus_size in CORPUS_SIZES:

    print(f"\n===== Corpus: {corpus_size:,} =====")

    corpus_vectors = np.ascontiguousarray(
        document_embeddings[:corpus_size],
        dtype=np.float32
    )

    index = faiss.IndexFlatIP(dimension)
    index.add(corpus_vectors)

    start = time.perf_counter()

    scores, indices = index.search(
        query_embeddings,
        TOP_K
    )

    elapsed = time.perf_counter() - start
    latency_ms = elapsed / len(query_embeddings) * 1000

    np.save(
        OUTPUT_DIR / f"nq_flat_{corpus_size}_top10_indices.npy",
        indices
    )

    np.save(
        OUTPUT_DIR / f"nq_flat_{corpus_size}_top10_scores.npy",
        scores
    )

    print("Total Search Time:", round(elapsed, 4), "s")
    print("Average Latency:", round(latency_ms, 4), "ms/query")
    print("Result Shape:", indices.shape)

print("\nFlatIP Reference Complete")
