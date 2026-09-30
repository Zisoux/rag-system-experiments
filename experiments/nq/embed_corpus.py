import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


########## 설정 ##########

MODEL_NAME = "BAAI/bge-m3"

TOTAL_DOCUMENTS = 500_000
EMBEDDING_DIMENSION = 1024

CHUNK_SIZE = 10_000
BATCH_SIZE = 16

EMBEDDING_PATH = Path("data/nq_corpus_500k_embeddings.npy")
PROGRESS_PATH = Path("data/nq_embedding_progress.json")


########## Embedding Model ##########

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Embedding Device:", device)

if device != "cuda":
    raise RuntimeError("CUDA를 사용할 수 없습니다.")

model = SentenceTransformer(
    MODEL_NAME,
    device=device
)


def embed_documents(documents):
    return model.encode(
        documents,
        normalize_embeddings=True,
        batch_size=BATCH_SIZE,
        show_progress_bar=True
    )


########## 데이터 로드 ##########

corpus_ids = np.load(
    "data/nq_corpus_500k_ids.npy"
)

corpus = load_dataset(
    "BeIR/nq",
    "corpus",
    split="corpus"
)

if len(corpus_ids) != TOTAL_DOCUMENTS:
    raise ValueError(
        f"Corpus ID 개수가 {TOTAL_DOCUMENTS}개가 아닙니다."
    )

print("")
print("===== NQ Corpus =====")
print("Target Corpus:", len(corpus_ids))
print("Full NQ Corpus:", len(corpus))


########## Target Document 추출 ##########

print("")
print("===== Target Documents Loading =====")

start_loading = time.perf_counter()

target_ids = set(corpus_ids)
document_map = {}

for document in corpus:
    document_id = document["_id"]

    if document_id in target_ids:
        document_map[document_id] = (
            document["title"]
            + "\n"
            + document["text"]
        )

    if len(document_map) == TOTAL_DOCUMENTS:
        break

if len(document_map) != TOTAL_DOCUMENTS:
    raise ValueError(
        f"500K 문서를 모두 찾지 못했습니다: {len(document_map)}"
    )

documents = [
    document_map[document_id]
    for document_id in corpus_ids
]

end_loading = time.perf_counter()

print("Loaded Documents:", len(documents))
print(
    "Document Loading Time:",
    end_loading - start_loading,
    "seconds"
)


########## 진행 상태 확인 ##########

if PROGRESS_PATH.exists():
    with open(
        PROGRESS_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        progress = json.load(file)

    start_index = progress["completed"]

    print("")
    print("===== Resume =====")
    print("Completed:", start_index)

else:
    start_index = 0


########## Embedding 저장 공간 생성 ##########

if EMBEDDING_PATH.exists():
    embeddings = np.lib.format.open_memmap(
        EMBEDDING_PATH,
        mode="r+"
    )

else:
    embeddings = np.lib.format.open_memmap(
        EMBEDDING_PATH,
        mode="w+",
        dtype=np.float32,
        shape=(
            TOTAL_DOCUMENTS,
            EMBEDDING_DIMENSION
        )
    )


########## 500K Embedding ##########

print("")
print("===== NQ 500K Embedding Start =====")

total_start = time.perf_counter()

for start in range(
    start_index,
    TOTAL_DOCUMENTS,
    CHUNK_SIZE
):
    end = min(
        start + CHUNK_SIZE,
        TOTAL_DOCUMENTS
    )

    print("")
    print(
        f"Embedding Documents: "
        f"{start:,} ~ {end:,}"
    )

    chunk_documents = documents[start:end]

    chunk_start = time.perf_counter()

    chunk_embeddings = embed_documents(
        chunk_documents
    )

    chunk_end = time.perf_counter()

    if chunk_embeddings.shape[1] != EMBEDDING_DIMENSION:
        raise ValueError(
            "Embedding Dimension이 예상과 다릅니다."
        )

    embeddings[start:end] = chunk_embeddings

    # 디스크에 즉시 반영
    embeddings.flush()

    # 현재 완료 위치 저장
    with open(
        PROGRESS_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            {"completed": end},
            file
        )

    chunk_time = chunk_end - chunk_start

    print(
        f"Chunk Completed: {end:,} / "
        f"{TOTAL_DOCUMENTS:,}"
    )
    print(
        "Chunk Time:",
        chunk_time,
        "seconds"
    )


########## 완료 ##########

total_end = time.perf_counter()

print("")
print("===== NQ 500K Embedding Complete =====")
print("Embedding Shape:", embeddings.shape)
print(
    "Total Execution Time:",
    total_end - total_start,
    "seconds"
)

if PROGRESS_PATH.exists():
    PROGRESS_PATH.unlink()

print("Embedding 저장 완료:", EMBEDDING_PATH)