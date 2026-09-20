import pandas as pd
import matplotlib.pyplot as plt

results = pd.read_csv("data/ivf_results.csv")
flat_result = pd.read_csv("data/flat_results.csv")

# nlist_10 = results[results["nlist"] == 10]

# plt.plot(
#     nlist_10["nprobe"],
#     nlist_10["recall_at_10"],
#     marker="o"
# )

nlist_values = [10, 25, 50, 100]

# 그래프 1 <IVF nprobe vs Recall@10>
for nlist in nlist_values:
    current_results = results[results["nlist"] == nlist]

    plt.plot(
        current_results["nprobe"],
        current_results["recall_at_10"],
        marker="o",
        label=f"nlist={nlist}"
    )

plt.xlabel("nprobe")
plt.ylabel("Recall@10")
plt.title("IVF nprobe vs Recall@10")
plt.legend()
plt.grid(True)

plt.show()

# 그래프 2 <IVF nprobe vs Latency>
for nlist in nlist_values:
    current_results = results[results["nlist"] == nlist]

    plt.plot(
        current_results["nprobe"],
        current_results["average_latency"],
        marker="o",
        label=f"nlist={nlist}"
    )

plt.xlabel("nprobe")
plt.ylabel("Average Latency (ms)")
plt.title("IVF nprobe vs Latency")
plt.legend()
plt.grid(True)

plt.show()

# 그래프 3 <IVF Accuracy-Latency Trade-off>
flat_latency = flat_result["average_latency"].iloc[0]
flat_recall = flat_result["recall_at_10"].iloc[0]

for nlist in nlist_values:
    current_results = results[results["nlist"] == nlist]

    plt.plot(
        current_results["average_latency"],
        current_results["recall_at_10"],
        marker="o",
        label=f"nlist={nlist}"
    )

plt.scatter(
    flat_latency,
    flat_recall,
    marker="*",
    s=200,
    label="FlatIP"
)

plt.xlabel("Average Latency (ms)")
plt.ylabel("Recall@10")
plt.title("IVF Accuracy-Latency Trade-off")
plt.legend()
plt.grid(True)

plt.show()