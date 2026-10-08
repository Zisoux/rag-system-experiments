import time

import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


########## 설정 ##########

MODEL_NAME = "BAAI/bge-m3"

QUERY_IDS_PATH = "data/nq_test_query_ids.npy"
OUTPUT_PATH = "data/nq_test_query_embeddings.npy"

BATCH_SIZE = 16


########## Embedding Model ##########

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Embedding Device:", device)

if device != "cuda":
    raise RuntimeError("CUDA를 사용할 수 없습니다.")

model = SentenceTransformer(
    MODEL_NAME,
    device=device
)


########## Query ID 로드 ##########

query_ids = np.load(
    QUERY_IDS_PATH
)

print("")
print("===== Test Query IDs =====")
print("Queries:", len(query_ids))


########## NQ Queries 로드 ##########

queries = load_dataset(
    "BeIR/nq",
    "queries",
    split="queries"
)

query_id_set = set(query_ids)

query_map = {}

for query in queries:
    query_id = query["_id"]

    if query_id in query_id_set:
        query_map[query_id] = query["text"]

    if len(query_map) == len(query_ids):
        break


if len(query_map) != len(query_ids):
    raise ValueError(
        f"일부 Query를 찾지 못했습니다: "
        f"{len(query_map)} / {len(query_ids)}"
    )


########## Query ID 순서에 맞춰 정렬 ##########

query_texts = [
    query_map[query_id]
    for query_id in query_ids
]


print("")
print("===== Query Check =====")
print("Queries:", len(query_texts))
print("First ID:", query_ids[0])
print("First Query:", query_texts[0])


########## Query Embedding ##########

print("")
print("===== Query Embedding Start =====")

start = time.perf_counter()

query_embeddings = model.encode(
    query_texts,
    normalize_embeddings=True,
    batch_size=BATCH_SIZE,
    show_progress_bar=True
)

end = time.perf_counter()


########## 검증 ##########

print("")
print("===== Query Embedding Result =====")
print("Shape:", query_embeddings.shape)
print(
    "Embedding Time:",
    round(end - start, 2),
    "seconds"
)

print(
    "First Vector Norm:",
    np.linalg.norm(query_embeddings[0])
)


########## 저장 ##########

np.save(
    OUTPUT_PATH,
    query_embeddings.astype(np.float32)
)

print("")
print(
    "Query Embeddings Saved:",
    OUTPUT_PATH
)