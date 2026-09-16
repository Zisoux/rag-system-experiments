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


## 너무 오래걸려서 대안 찾자. 0917에 계속...
document_embeddings = embed_documents(documents)
print("document_embeddings 개수: ",len(document_embeddings))