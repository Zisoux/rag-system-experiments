import time

import faiss
import numpy as np

EMBEDDING_PATH = "data/nq_corpus_500k_embeddings.npy"

NLIST_CANDIDATES = {
    10_000: [128, 256],
    30_000: [256, 512],
    50_000: [512, 1024],
}

TRAIN_SEED = 42

embeddings = np.load(
    EMBEDDING_PATH,
    mmap_mode="r"
)

dimension = embeddings.shape[1]

print("===== Small Corpus nlist Pilot =====")

for corpus_size, candidates in NLIST_CANDIDATES.items():

    vectors = np.ascontiguousarray(
        embeddings[:corpus_size],
        dtype=np.float32
    )

    for nlist in candidates:

        print(f"\n===== Corpus: {corpus_size:,} | nlist: {nlist} =====")

        quantizer = faiss.IndexFlatIP(dimension)

        index = faiss.IndexIVFFlat(
            quantizer,
            dimension,
            nlist,
            faiss.METRIC_INNER_PRODUCT
        )

        index.cp.seed = TRAIN_SEED

        start = time.perf_counter()
        index.train(vectors)
        train_time = time.perf_counter() - start

        start = time.perf_counter()
        index.add(vectors)
        add_time = time.perf_counter() - start

        list_sizes = np.array([
            index.invlists.list_size(i)
            for i in range(nlist)
        ])

        print("Train Time:", round(train_time, 2), "s")
        print("Add Time:", round(add_time, 2), "s")
        print("Avg Vectors/Cluster:", round(list_sizes.mean(), 2))
        print("Min Vectors/Cluster:", list_sizes.min())
        print("Max Vectors/Cluster:", list_sizes.max())
        print("Empty Clusters:", np.sum(list_sizes == 0))
