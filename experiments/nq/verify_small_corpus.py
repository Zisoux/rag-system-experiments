import numpy as np
from datasets import load_dataset

CORPUS_SIZES = [10_000, 30_000, 50_000]

corpus_ids = np.load("data/nq_corpus_500k_ids.npy")

embeddings = np.load(
    "data/nq_corpus_500k_embeddings.npy",
    mmap_mode="r"
)

qrels = load_dataset(
    "BeIR/nq-qrels",
    split="test"
)

relevant_ids = {
    row["corpus-id"]
    for row in qrels
    if row["score"] > 0
}

print("===== Small Corpus Verification =====")
print("Corpus IDs:", len(corpus_ids))
print("Embedding Shape:", embeddings.shape)
print("Relevant Documents:", len(relevant_ids))

assert len(corpus_ids) == len(embeddings)
assert len(set(corpus_ids.tolist())) == len(corpus_ids)

for size in CORPUS_SIZES:
    current_ids = set(corpus_ids[:size].tolist())
    missing = relevant_ids - current_ids

    print(f"\n===== {size:,} Corpus =====")
    print("Relevant Included:", len(missing) == 0)
    print("Missing Documents:", len(missing))
    print("Embedding Shape:", embeddings[:size].shape)
