import statistics

from src.rag_pipeline import run_rag


QUERY ="Astra는 해킹 능력이 어느 정도야?"
TOP_K = 3
RUNS = 10


embedding_times = []
retrieval_times = []
generation_times = []
end_to_end_times = []



for run in range(1, RUNS + 1):
    print(f"===== Run {run} =====")
    answer, results, metrics = run_rag(query=QUERY, k=TOP_K)

    embedding_times.append(metrics["embedding_ms"])
    retrieval_times.append(metrics["retrieval_ms"])
    generation_times.append(metrics["generation_ms"])
    end_to_end_times.append(metrics["end_to_end_ms"])

    print(f"Embedding: {metrics['embedding_ms']:.2f} ms")
    print(f"Retrieval: {metrics['retrieval_ms']:.2f} ms")
    print(f"Generation: {metrics['generation_ms']:.2f} ms")
    print(f"End-to-End: {metrics['end_to_end_ms']:.2f} ms")
    print()


def print_statistics(name, values):
    print(f"{name}")
    print(f"  Average: {statistics.mean(values):.2f} ms")
    print(f"  Minimum: {min(values):.2f} ms")
    print(f"  Maximum: {max(values):.2f} ms")
    print()


print("===== Baseline Latency Summary =====")
print(f"Query: {QUERY}")
print(f"Top K: {TOP_K}")
print(f"Runs: {RUNS}")
print()


print_statistics("Query Embedding", embedding_times)
print_statistics("FAISS Retrieval", retrieval_times)
print_statistics("Generation", generation_times)
print_statistics("End-to-End", end_to_end_times)