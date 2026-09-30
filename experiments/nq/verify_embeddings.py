import numpy as np


embeddings = np.load(
    "data/nq_corpus_500k_embeddings.npy",
    mmap_mode="r"
)

print("===== NQ Embedding Verification =====")
print("Shape:", embeddings.shape)
print("Dtype:", embeddings.dtype)

print("NaN:", np.isnan(embeddings).any())
print("Inf:", np.isinf(embeddings).any())

print(
    "First Vector Norm:",
    np.linalg.norm(embeddings[0])
)