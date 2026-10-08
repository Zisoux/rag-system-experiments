import csv
import time
from pathlib import Path

import faiss
import numpy as np
from datasets import load_dataset


########## 설정 ##########

DOCUMENT_EMBEDDING_PATH = "data/nq_corpus_500k_embeddings.npy"
QUERY_EMBEDDING_PATH = "data/nq_test_query_embeddings.npy"

CORPUS_ID_PATH = "data/nq_corpus_500k_ids.npy"
QUERY_ID_PATH = "data/nq_test_query_ids.npy"

RESULT_PATH = "data/nq_ivf_small_results.csv"

TOP_K = 10
TRAIN_SEED = 42

NLIST = {
    10_000: 128,
    30_000: 256,
    50_000: 512,
}

NPROBE_VALUES = [
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128
]


########## 데이터 로드 ##########

document_embeddings = np.load(
    DOCUMENT_EMBEDDING_PATH,
    mmap_mode="r"
)

query_embeddings = np.ascontiguousarray(
    np.load(QUERY_EMBEDDING_PATH),
    dtype=np.float32
)

corpus_ids = np.load(CORPUS_ID_PATH)
query_ids = np.load(QUERY_ID_PATH)

dimension = document_embeddings.shape[1]

print("===== NQ Small Corpus IVF Experiment =====")
print("Document Embeddings:", document_embeddings.shape)
print("Query Embeddings:", query_embeddings.shape)
print("Corpus IDs:", len(corpus_ids))
print("Query IDs:", len(query_ids))
print("Dimension:", dimension)

if len(document_embeddings) != len(corpus_ids):
    raise ValueError("Document Embedding과 Corpus ID 개수가 다릅니다.")

if len(query_embeddings) != len(query_ids):
    raise ValueError("Query Embedding과 Query ID 개수가 다릅니다.")

if query_embeddings.shape[1] != dimension:
    raise ValueError("Document와 Query의 Embedding 차원이 다릅니다.")

if len(corpus_ids) < max(NLIST):
    raise ValueError("Corpus 데이터가 부족합니다.")


########## Qrels 로드 ##########

qrels_dataset = load_dataset(
    "BeIR/nq-qrels",
    split="test"
)

qrels = {}

for row in qrels_dataset:

    if row["score"] <= 0:
        continue

    query_id = str(row["query-id"])
    document_id = str(row["corpus-id"])

    qrels.setdefault(
        query_id,
        set()
    ).add(document_id)

print("Qrels Queries:", len(qrels))

missing_query_ids = set(
    map(str, query_ids)
) - set(qrels)

if missing_query_ids:
    raise ValueError(
        f"Qrels에 없는 Query ID가 있습니다: {len(missing_query_ids)}"
    )


########## Recall@10 ##########

def calculate_recall(search_indices, current_corpus_ids):

    recalls = []

    for query_position, result_indices in enumerate(search_indices):

        query_id = str(query_ids[query_position])

        relevant_documents = qrels.get(
            query_id,
            set()
        )

        if not relevant_documents:
            continue

        retrieved_documents = {
            str(current_corpus_ids[index])
            for index in result_indices
            if index >= 0
        }

        found = len(
            relevant_documents & retrieved_documents
        )

        recall = found / len(relevant_documents)
        recalls.append(recall)

    return float(np.mean(recalls))


########## FlatIP Overlap@10 ##########

def calculate_overlap(ivf_indices, flat_indices):

    overlaps = []

    for ivf_result, flat_result in zip(
        ivf_indices,
        flat_indices
    ):

        ivf_set = {
            int(index)
            for index in ivf_result
            if index >= 0
        }

        flat_set = {
            int(index)
            for index in flat_result
            if index >= 0
        }

        overlap = len(
            ivf_set & flat_set
        ) / TOP_K

        overlaps.append(overlap)

    return float(np.mean(overlaps))


########## Scanned Vectors ##########

def calculate_scanned_vectors(index, query_vectors, nprobe):

    _, list_indices = index.quantizer.search(
        query_vectors,
        nprobe
    )

    list_sizes = np.array(
        [
            index.invlists.list_size(i)
            for i in range(index.nlist)
        ],
        dtype=np.int64
    )

    scanned_counts = []

    for query_lists in list_indices:

        scanned = sum(
            int(list_sizes[list_id])
            for list_id in query_lists
            if list_id >= 0
        )

        scanned_counts.append(scanned)

    return float(np.mean(scanned_counts))


########## 결과 저장 준비 ##########

results = []

Path(RESULT_PATH).parent.mkdir(
    parents=True,
    exist_ok=True
)


########## Corpus Size별 실험 ##########

for corpus_size, nlist in NLIST.items():

    print("")
    print("=" * 70)
    print(
        f"Corpus Size: {corpus_size:,} "
        f"| nlist: {nlist}"
    )
    print("=" * 70)


    ########## Corpus 준비 ##########

    corpus_vectors = np.ascontiguousarray(
        document_embeddings[:corpus_size],
        dtype=np.float32
    )

    current_corpus_ids = corpus_ids[:corpus_size]

    relevant_document_ids = set().union(
        *qrels.values()
    )

    missing_documents = (
        relevant_document_ids
        - set(map(str, current_corpus_ids))
    )

    if missing_documents:
        raise ValueError(
            f"{corpus_size:,} Corpus에 정답 문서 "
            f"{len(missing_documents)}개가 누락됐습니다."
        )


    ########## FlatIP Reference 로드 ##########

    flat_path = (
        f"data/nq_flat_{corpus_size}_top10_indices.npy"
    )

    flat_indices = np.load(flat_path)

    expected_shape = (
        len(query_embeddings),
        TOP_K
    )

    if flat_indices.shape != expected_shape:
        raise ValueError(
            f"FlatIP 결과 Shape 오류: {flat_indices.shape}"
        )


    ########## IVF Index 생성 ##########

    quantizer = faiss.IndexFlatIP(dimension)

    index = faiss.IndexIVFFlat(
        quantizer,
        dimension,
        nlist,
        faiss.METRIC_INNER_PRODUCT
    )

    index.cp.seed = TRAIN_SEED


    ########## Train ##########

    print("")
    print("Training IVF...")

    train_start = time.perf_counter()

    index.train(corpus_vectors)

    train_time = time.perf_counter() - train_start

    print(
        "Train Time:",
        round(train_time, 4),
        "seconds"
    )


    ########## Add ##########

    print("Adding vectors...")

    add_start = time.perf_counter()

    index.add(corpus_vectors)

    add_time = time.perf_counter() - add_start

    print(
        "Add Time:",
        round(add_time, 4),
        "seconds"
    )

    if index.ntotal != corpus_size:
        raise ValueError("IVF Index에 저장된 벡터 수가 다릅니다.")


    ########## nprobe 실험 ##########

    for nprobe in NPROBE_VALUES:

        if nprobe > nlist:
            continue

        index.nprobe = nprobe

        print("")
        print(f"----- nprobe = {nprobe} -----")


        ########## Search ##########

        search_start = time.perf_counter()

        _, ivf_indices = index.search(
            query_embeddings,
            TOP_K
        )

        total_search_time = (
            time.perf_counter() - search_start
        )

        latency_ms = (
            total_search_time
            / len(query_embeddings)
            * 1000
        )


        ########## Recall@10 ##########

        recall = calculate_recall(
            ivf_indices,
            current_corpus_ids
        )


        ########## FlatIP Overlap@10 ##########

        overlap = calculate_overlap(
            ivf_indices,
            flat_indices
        )


        ########## Scanned Vectors ##########

        scanned_vectors = calculate_scanned_vectors(
            index,
            query_embeddings,
            nprobe
        )


        ########## 결과 출력 ##########

        print("Recall@10:", round(recall, 4))

        print(
            "Flat Top-10 Overlap:",
            round(overlap, 4)
        )

        print(
            "Average Latency:",
            round(latency_ms, 4),
            "ms/query"
        )

        print(
            "Average Scanned Vectors:",
            round(scanned_vectors, 2)
        )


        ########## 결과 기록 ##########

        results.append({
            "corpus_size": corpus_size,
            "nlist": nlist,
            "nprobe": nprobe,
            "recall_at_10": recall,
            "flat_overlap_at_10": overlap,
            "latency_ms": latency_ms,
            "avg_scanned_vectors": scanned_vectors
        })


########## CSV 저장 ##########

with open(
    RESULT_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "corpus_size",
            "nlist",
            "nprobe",
            "recall_at_10",
            "flat_overlap_at_10",
            "latency_ms",
            "avg_scanned_vectors"
        ]
    )

    writer.writeheader()
    writer.writerows(results)


print("")
print("=" * 70)
print("Experiment Complete")
print("Results Saved:", RESULT_PATH)
print("=" * 70)
