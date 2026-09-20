import os
import numpy as np

from datasets import load_dataset
from src.embedding import embed_documents

corpus = load_dataset(
    "BeIR/scifact",
    "corpus",
    split="corpus"
)

print("1. Corpus 총 개수: ", len(corpus))
print("2. 첫 번째 Corpus Data: ", corpus[0] )


queries = load_dataset(
    "BeIR/scifact",
    "queries",
    split="queries"
)

print()
print("1. Query 전채 개수: ", len(queries))
print("2. 첫 번째 Query: ", queries[0])


qrels = load_dataset(
    "BeIR/scifact-qrels",
    split="test"
)

print()
print("1. Qrels 전채 개수: ", len(qrels))
print("2. 첫 번째 Qrels Data:", qrels[0])
print()


target_query_id = str(qrels[0]["query-id"])

for query in queries:
    if query["_id"] == target_query_id:
        print("Query: ", query)

print()

for cor in corpus:
    if cor["_id"] == "31715818":
        print("Corpus: ", cor)



documents = []

for doc in corpus:
    document = doc["title"] + " " + doc["text"]
    documents.append(document)

document_ids = []

for docu in corpus:
    document_id = docu["_id"]
    document_ids.append(document_id)


document_embeddings = embed_documents(documents)

embedding_path = "data/scifact_embeddings.npy"
document_ids_path = "data/scifact_document_ids.npy"

np.save(embedding_path, document_embeddings)
np.save(document_ids_path, np.array(document_ids))

print("Embedding 저장 완료\n")
print("Embedding 파일:", os.path.abspath(embedding_path))
print("Document ID 파일:", os.path.abspath(document_ids_path))

print("Embedding 개수: ",len(document_embeddings))
print("Embedding Dimension: ", document_embeddings.shape[1])