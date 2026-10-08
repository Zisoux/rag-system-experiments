from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULT_PATH = Path("data/nq_ivf_small_results.csv")
FIGURE_DIR = Path("experiments/nq/figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RESULT_PATH)
corpus_sizes = sorted(df["corpus_size"].unique())

def plot_metric(x, y, xlabel, ylabel, title, filename):
    plt.figure(figsize=(8, 5))

    for size in corpus_sizes:
        subset = df[df["corpus_size"] == size].sort_values("nprobe")
        plt.plot(
            subset[x],
            subset[y],
            marker="o",
            label=f"{size // 1000}K"
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(title="Corpus Size")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=300)
    plt.close()

plot_metric(
    "nprobe",
    "recall_at_10",
    "nprobe",
    "Recall@10",
    "nprobe vs Recall@10",
    "nprobe_recall.png"
)

plot_metric(
    "nprobe",
    "latency_ms",
    "nprobe",
    "Latency (ms/query)",
    "nprobe vs Retrieval Latency",
    "nprobe_latency.png"
)

plot_metric(
    "latency_ms",
    "recall_at_10",
    "Latency (ms/query)",
    "Recall@10",
    "Quality-Latency Trade-off",
    "accuracy_latency_tradeoff.png"
)

print("NQ experiment figures saved:", FIGURE_DIR)
