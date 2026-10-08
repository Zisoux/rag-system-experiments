import random
import numpy as np

from datasets import load_dataset



########## NQ 데이터셋 로드 ##########

corpus = load_dataset(
    "BeIR/nq",
    "corpus",
    split="corpus"
)

queries = load_dataset(
    "BeIR/nq",
    "queries",
    split="queries"
)

qrels = load_dataset(
    "BeIR/nq-qrels",
    split="test"
)


print("===== NQ Dataset =====")
print("Corpus:", len(corpus))
print("Queries:", len(queries))
print("Qrels:", len(qrels))

print("")
print("===== Corpus Sample =====")
print(corpus[0])

print("")
print("===== Query Sample =====")
print(queries[0])

print("")
print("===== Qrels Sample =====")
print(qrels[0])



########## Test Qrels 확인 ##########
# Test qrels에 포함된 Query ID
test_query_ids = set()

# Test qrels에 포함된 Relevant Document ID
relevant_document_ids = set()

for qrel in qrels:
    test_query_ids.add(qrel["query-id"])
    relevant_document_ids.add(qrel["corpus-id"])


print("")
print("===== Test Qrels =====")
print("Unique Test Queries:", len(test_query_ids))
print("Unique Relevant Documents:", len(relevant_document_ids))



########## Nested Corpus 구성 ##########

random_seed = 42
random.seed(random_seed)

# Relevant Document를 제외한 Distractor 후보
distractor_ids = []

for document in corpus:
    document_id = document["_id"]

    if document_id not in relevant_document_ids:
        distractor_ids.append(document_id)


print("")
print("===== Distractor Candidates =====")
print("Distractor Candidates:", len(distractor_ids))


# Random Seed를 고정한 상태로 Distractor 순서 섞기
random.shuffle(distractor_ids)

target_corpus_size = 500_000

required_distractor_count = (
    target_corpus_size - len(relevant_document_ids)
)

selected_distractor_ids = distractor_ids[
    :required_distractor_count
]


# Relevant Document + Random Distractor
corpus_500k_ids = (
    sorted(relevant_document_ids)
    + selected_distractor_ids
)


# Nested Corpus
corpus_100k_ids = corpus_500k_ids[:100_000]
corpus_300k_ids = corpus_500k_ids[:300_000]
corpus_500k_ids = corpus_500k_ids[:500_000]


print("")
print("===== Nested Corpus =====")
print("100K:", len(corpus_100k_ids))
print("300K:", len(corpus_300k_ids))
print("500K:", len(corpus_500k_ids))



########## Relevant Document 포함 여부 확인 ##########

print("")
print("===== Relevant Document Check =====")

print(
    "100K:",
    relevant_document_ids.issubset(set(corpus_100k_ids))
)

print(
    "300K:",
    relevant_document_ids.issubset(set(corpus_300k_ids))
)

print(
    "500K:",
    relevant_document_ids.issubset(set(corpus_500k_ids))
)



########## 실험 데이터 ID 저장 ##########

np.save(
    "data/nq_corpus_500k_ids.npy",
    np.array(corpus_500k_ids)
)

np.save(
    "data/nq_test_query_ids.npy",
    np.array(sorted(test_query_ids))
)

print("")
print("NQ Corpus / Query ID 저장 완료")