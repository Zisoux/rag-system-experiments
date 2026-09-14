import time
import faiss
from sentence_transformers import SentenceTransformer
from chunking import load_document, split_into_chunks



# 1. 문서 불러오기 + chunking
document = load_document("data/sample.txt")
chunks = split_into_chunks(document)


# 2. Embedding Model 불러오기
model = SentenceTransformer("BAAI/bge-m3")


# 3. Chunk Embedding 
embeddings = model.encode(
    chunks,
    normalize_embeddings=True # 각 벡터 길이 1로 맞춰주는 작업
)


# 4. FAISS Index 생성
dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension) # 정규화 된 벡터끼리는 내적(inner product) = 코사인 유사도


# 5. Vector 저장
index.add(embeddings)



def retrieve(query, k=3):

    # Query Embedding 시간 측정
    embedding_start = time.perf_counter()

    # 6. Query Embedding
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    embedding_end = time.perf_counter()

    # FAISS Retrieval 시간 측정
    retrieval_start = time.perf_counter()

    # 7. 유사한 청크 검색
    scores, indices = index.search(query_embedding, k)

    retrieval_end = time.perf_counter()


    results = []

    for rank, result in enumerate(
        zip(scores[0], indices[0]),
        start=1
    ):
        score, idx = result

        results.append({
            "rank": rank,
            "score": float(score),
            "chunk_index": int(idx),
            "text": chunks[idx]
        })

    retrieved_chunks = []

    for result in results:
        retrieved_chunks.append(result["text"])

    context = "\n\n".join(retrieved_chunks)


    metrics = {
        "embedding_ms": (embedding_end - embedding_start) * 1000,
        "retrieval_ms": (retrieval_end - retrieval_start) * 1000
    }


    return context, results, metrics
