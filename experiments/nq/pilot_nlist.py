import time

import faiss
import numpy as np


########## 설정 ##########

EMBEDDING_PATH = "data/nq_corpus_500k_embeddings.npy"
DIMENSION = 1024

EXPERIMENTS = {
    100_000: [1024, 2048],
    300_000: [2048, 4096],
    500_000: [4096, 8192],
}


########## Embedding 로드 ##########

all_embeddings = np.load(
    EMBEDDING_PATH,
    mmap_mode="r"
)

print("===== NQ nlist Pilot =====")
print("Embedding Shape:", all_embeddings.shape)


########## Pilot ##########

for corpus_size, nlist_candidates in EXPERIMENTS.items():

    print("")
    print("=" * 60)
    print(f"Corpus Size: {corpus_size:,}")
    print("=" * 60)

    embeddings = np.asarray(
        all_embeddings[:corpus_size],
        dtype=np.float32
    )

    for nlist in nlist_candidates:

        print("")
        print(f"----- nlist = {nlist} -----")

        quantizer = faiss.IndexFlatIP(DIMENSION)

        index = faiss.IndexIVFFlat(
            quantizer,
            DIMENSION,
            nlist,
            faiss.METRIC_INNER_PRODUCT
        )

        ########## Train ##########

        train_start = time.perf_counter()

        index.train(embeddings)

        train_end = time.perf_counter()

        ########## Add ##########

        add_start = time.perf_counter()

        index.add(embeddings)

        add_end = time.perf_counter()

        ########## Cluster 분포 ##########

        list_sizes = np.array([
            index.invlists.list_size(i)
            for i in range(nlist)
        ])

        print(
            "Train Time:",
            round(train_end - train_start, 2),
            "seconds"
        )

        print(
            "Add Time:",
            round(add_end - add_start, 2),
            "seconds"
        )

        print(
            "Average Vectors / Cluster:",
            round(list_sizes.mean(), 2)
        )

        print(
            "Minimum Vectors / Cluster:",
            list_sizes.min()
        )

        print(
            "Maximum Vectors / Cluster:",
            list_sizes.max()
        )

        print(
            "Empty Clusters:",
            np.sum(list_sizes == 0)
        )