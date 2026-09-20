from sentence_transformers import SentenceTransformer
from src.chunking import load_document, split_into_chunks


# document = load_document("data/sample.txt")
# chunks = split_into_chunks(document)

# 고정된 연구실 환경이 없다고 하셔서 한국어 포함 다국어 지원되는 범용 모델 선택했습니다.
model = SentenceTransformer("BAAI/bge-m3")

# embeddings = model.encode(chunks)

# print("Chunk 개수:", len(chunks))
# print("Embedding 개수:", len(embeddings))
# print("Embedding Dimension:", embeddings.shape[1])



def embed_documents(documents):
    embeddings = model.encode(
        documents,
        normalize_embeddings=True,
        batch_size=16,
        show_progress_bar=True
    )

    return embeddings
