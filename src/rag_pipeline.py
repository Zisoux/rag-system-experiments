import os
import time

from dotenv import load_dotenv
from openai import OpenAI

from src.retrieval import retrieve


load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)



def run_rag(query, k=3):
    total_start = time.perf_counter()

    context, results, metrics = retrieve(query=query, k=k)


    prompt = f"""
    다음 Context에 있는 정보만 참고해서 질문에 답하세요.
    Context에 답이 없다면 모른다고 답하세요.

    [Context]
    {context}

    [Question]
    {query}
    """

    generation_start = time.perf_counter()

    response = client.responses.create(
        model="openai/gpt-oss-20b",
        input=prompt
    )

    generation_end = time.perf_counter()

    total_end = time.perf_counter()

    # 현재는 Groq API를 사용하고 있어 API 요청부터 전체 응답까지의 시간입니다!
    metrics["generation_ms"] = (generation_end - generation_start) * 1000
    metrics["end_to_end_ms"] = (total_end - total_start) * 1000

    return response.output_text, results, metrics



if __name__ == "__main__":
    query = "Astra는 해킹 능력이 어느 정도야?"

    answer, results, metrics = run_rag(query=query, k=3)

    print("===== Retrieval Results =====")
    for result in results:
        print(f"[Rank {result['rank']}]")
        print(f"score: {result['score']:.4f})")
        print(result['text'])
        print()


    print("===== Performance =====")
    print(f"Query Embedding Time: {metrics['embedding_ms']:.2f} ms")
    print(f"FAISS Retrieval Time: {metrics['retrieval_ms']:.2f} ms")
    print(f"Generation Time: {metrics['generation_ms']:.2f} ms")
    print(f"End-to-End Latency: {metrics['end_to_end_ms']:.2f} ms")

    print()
    print("===== Final Answer =====")
    print(answer)



