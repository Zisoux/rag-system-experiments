import csv
import time
import numpy as np
import faiss

from datasets import load_dataset
# from src.embedding import embed_documents


########## FlatIP 실험 ##########
document_embeddings = np.load("data/scifact_embeddings.npy")
document_ids = np.load("data/scifact_document_ids.npy")

print("Document Embeddings: ", len(document_embeddings))
print("Document IDs: ", len(document_ids))
print("Embedding Shape: ", document_embeddings.shape)


dimension = document_embeddings.shape[1]

flat_index = faiss.IndexFlatIP(dimension)
flat_index.add(document_embeddings)

print("")
print("Flat Index Vector Count: ", flat_index.ntotal)


# queries = load_dataset(
#     "BeIR/scifact",
#     "queries",
#     split="queries"
# )

qrels = load_dataset(
    "BeIR/scifact-qrels",
    split="test"
)

# test_query_ids = set()

# for qrel in qrels:
#     test_query_ids.add(qrel["query-id"])

# print("")
# print("Test Query Count: ", len(test_query_ids))


# test_queries = []
# test_query_id_list = []

# for query in queries:
#     query_id = int(query["_id"])

#     if query_id in test_query_ids:
#         test_queries.append(query["text"])
#         test_query_id_list.append(query_id)
    

# print("")
# print(len(test_queries))
# print(len(test_query_id_list))


# query_embeddings = embed_documents(test_queries)

# np.save("data/scifact_query_embeddings.npy", query_embeddings)
# np.save("data/scifact_query_ids.npy", np.array(test_query_id_list))

query_embeddings = np.load("data/scifact_query_embeddings.npy")
query_ids = np.load("data/scifact_query_ids.npy")

top_k = 10


flat_latencies = []
runs = 10

for i in range(runs):
    start = time.perf_counter()

    scores, indices = flat_index.search(
        query_embeddings,
        top_k
    )

    end = time.perf_counter()

    flat_latency_ms = (end - start) * 1000
    flat_latencies.append(flat_latency_ms)

flat_average = sum(flat_latencies) / len(flat_latencies)
flat_minimum = min(flat_latencies)
flat_maximum = max(flat_latencies)

print("")
print("===== FlatIP Latency =====")
print("Runs:", runs)
print("Average:", flat_average, "ms")
print("Minimum:", flat_minimum, "ms")
print("Maximum:", flat_maximum, "ms")

print("")
print("Scores Shape:", scores.shape)
print("Indices Shape", indices.shape)


relevant_docs= {}

for qrel in qrels:
    query_id = qrel["query-id"]
    corpus_id = qrel["corpus-id"]

    if query_id not in relevant_docs:
        relevant_docs[query_id] = set()

    relevant_docs[query_id].add(corpus_id)

    
print("Relevant Query Count:", len(relevant_docs))


# first_result_indices = indices[0]

# retrieved_doc_ids = []

# for index in first_result_indices:
#     retrieved_doc_ids.append(int(document_ids[index]))


# first_query_id = query_ids[0]

# correct_docs = set(retrieved_doc_ids) & relevant_docs[first_query_id]

# first_recall = len(correct_docs) / len(relevant_docs[first_query_id])

# print("")
# print("First Query ID:", first_query_id)
# print("Retrieved:", retrieved_doc_ids)
# print("Relevant:", relevant_docs[first_query_id])
# print("Correct:", correct_docs)
# print("Recall@10:", first_recall)


recalls = []

for i in range(len(query_ids)):
    query_id = query_ids[i]
    result_indices = indices[i]

    retrieved_doc_ids = []

    for index in result_indices:
        retrieved_doc_ids.append(int(document_ids[index]))

    correct_docs = set(retrieved_doc_ids) & relevant_docs[query_id]

    recalls.append(len(correct_docs) / len(relevant_docs[query_id]))


recalls_average = sum(recalls) / len(recalls)

print("")
print("===== FlatIP Retrieval Accuracy =====")
print("Recall@10:", recalls_average)

flat_result = {
    "average_latency": flat_average,
    "minimum_latency": flat_minimum,
    "maximum_latency": flat_maximum,
    "recall_at_10": recalls_average
}

with open(
    "data/flat_results.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:

    fieldnames = [
        "average_latency",
        "minimum_latency",
        "maximum_latency",
        "recall_at_10"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerow(flat_result)

print("FlatIP 실험 결과 CSV 저장 완료")



########## IVF 실험 ##########
# 전체를 몇 개 Cluster로 나눌지
# nlist = 50
nlist_values = [10, 25, 50, 100]
base_nprobe_values = [1, 2, 4, 8, 16, 32, 64]

ivf_results = []

for nlist in nlist_values:
    quantizer = faiss.IndexFlatIP(dimension)

    ivf_index = faiss.IndexIVFFlat(
        quantizer, 
        dimension,
        nlist,
        faiss.METRIC_INNER_PRODUCT
    )

    ivf_index.train(document_embeddings)
    ivf_index.add(document_embeddings)

    nprobe_values = [
        value for value in base_nprobe_values
        if value < nlist
    ]

    nprobe_values.append(nlist)

# 만들어진 Cluster 중 Query와 가장 가까운 Cluster 수
# nprobe = 1
# ivf_index.nprobe = nprobe

# nprobe_values = [1, 2, 4, 8, 16, 32, 50]


    for nprobe in nprobe_values:
        ivf_index.nprobe = nprobe

        ivf_latencies = []
        runs = 10

        for i in range(runs):
            start = time.perf_counter()

            ivf_scores, ivf_indices = ivf_index.search(
                query_embeddings,
                top_k
            )   

            end = time.perf_counter()

            ivf_latency_ms = (end - start) * 1000
            ivf_latencies.append(ivf_latency_ms)


        ivf_average = sum(ivf_latencies) / len(ivf_latencies)
        ivf_minimum = min(ivf_latencies)
        ivf_maximum = max(ivf_latencies)

        print("")
        print("===== IVF Latency =====")
        print("nlist:", nlist)
        print("nprobe:", nprobe)
        print("Runs:", runs)
        print("Average:", ivf_average, "ms")
        print("Minimum:", ivf_minimum, "ms")
        print("Maximum:", ivf_maximum, "ms")


        ivf_recalls = []

        for i in range(len(query_ids)):
            query_id = query_ids[i]
            result_ivf_indices = ivf_indices[i]

            retrieved_doc_ids = []

            for index in result_ivf_indices:
                retrieved_doc_ids.append(int(document_ids[index]))

            correct_docs = set(retrieved_doc_ids) & relevant_docs[query_id]

            ivf_recalls.append(
                len(correct_docs) / len(relevant_docs[query_id])
            )


        ivf_recall_average = sum(ivf_recalls) / len(ivf_recalls)

        print("")
        print("===== IVF Retrieval Accuracy =====")
        print("nlist:", nlist)
        print("nprobe:", nprobe)
        print("Recall@10:", ivf_recall_average)

        result = {
            "nlist": nlist,
            "nprobe": nprobe,
            "average_latency": ivf_average,
            "minimum_latency": ivf_minimum,
            "maximum_latency": ivf_maximum,
            "recall_at_10": ivf_recall_average
        }

        ivf_results.append(result)


print("")
print("===== IVF Experiment Results =====")

for result in ivf_results:
    print(result)

with open(
    "data/ivf_results.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:

    fieldnames = [
        "nlist",
        "nprobe",
        "average_latency",
        "minimum_latency",
        "maximum_latency",
        "recall_at_10"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(ivf_results)

print("IVF 실험 결과 CSV 저장 완료")