# experiments/visualize_results.py

import json
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set(style="whitegrid")


# -----------------------------------------------------------
# Load metrics
# -----------------------------------------------------------
def load_metrics(file_path):
    with open(file_path, "r") as f:
        metrics = json.load(f)

    epochs = metrics.get("epochs", [])
    n = len(epochs)

    def pad(key):
        vals = metrics.get(key)
        if vals is None:
            return [float("nan")] * n
        vals = list(vals)
        if len(vals) < n:
            vals += [float("nan")] * (n - len(vals))
        return vals

    df = pd.DataFrame({
        "epoch": epochs,
        "loss": pad("train_loss") or pad("loss"),
        "accuracy": pad("accuracy"),
        "precision": pad("precision"),
        "recall": pad("recall"),
        "f1": pad("f1"),
        "epoch_time": pad("epoch_time")
    })

    return df, metrics


# -----------------------------------------------------------
# Metric plot
# -----------------------------------------------------------
def plot_metric(df, metric, ylabel, filename, out_dir):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="epoch", y=metric, marker="o")
    plt.title(f"{ylabel} Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.savefig(os.path.join(out_dir, f"{filename}.png"), dpi=300)
    plt.close()


# -----------------------------------------------------------
# Total training time
# -----------------------------------------------------------
def plot_total_time(metrics, out_dir, filename="total_training_time"):
    total_time = metrics.get("total_training_time") or metrics.get("total_time")
    if total_time is None:
        print("⚠️ No total training time found.")
        return

    plt.figure(figsize=(6, 6))
    sns.barplot(x=["Model"], y=[total_time])
    plt.title("Total Training Time")
    plt.ylabel("Time (seconds)")
    plt.savefig(os.path.join(out_dir, f"{filename}.png"), dpi=300)
    plt.close()


# -----------------------------------------------------------
# Confusion matrix (optional)
# -----------------------------------------------------------
def plot_confusion_matrix(cm, out_dir, filename="confusion_matrix"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(out_dir, f"{filename}.png"), dpi=300)
    plt.close()


# -----------------------------------------------------------
# Proper pipeline Gantt chart
# -----------------------------------------------------------
def plot_pipeline_gantt(events, out_dir, filename="pipeline_gantt"):
    if not events:
        print("[SKIP] Gantt chart: no pipeline events")
        return

    df = pd.DataFrame(events)

    # ensure required columns
    if not {"stage", "start", "end"}.issubset(df.columns):
        print("[WARN] Gantt chart: events missing stage/start/end")
        return

    df["duration"] = df["end"] - df["start"]
    df["start_rel"] = df["start"] - df["start"].min()

    plt.figure(figsize=(12, 6))

    stages = sorted(df["stage"].unique())
    for stage in stages:
        group = df[df["stage"] == stage]
        plt.barh(
            [f"Stage {stage}"] * len(group),
            group["duration"],
            left=group["start_rel"],
            alpha=0.7
        )

    plt.xlabel("Time (sec, relative)")
    plt.ylabel("Pipeline Stage")
    plt.title("Pipeline Gantt Chart")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{filename}.png"), dpi=300)
    plt.close()


# -----------------------------------------------------------
# MAIN entrypoint
# -----------------------------------------------------------
def generate_all_plots(file_path, mode):
    """
    mode: "sequential" or "parallel"
    """
    out_dir = os.path.join("../results/plots", mode)
    os.makedirs(out_dir, exist_ok=True)

    df, metrics = load_metrics(file_path)

    # core metrics
    plot_metric(df, "loss", "Training Loss", "loss_over_epochs", out_dir)
    plot_metric(df, "accuracy", "Accuracy", "accuracy_over_epochs", out_dir)
    plot_metric(df, "precision", "Precision", "precision_over_epochs", out_dir)
    plot_metric(df, "recall", "Recall", "recall_over_epochs", out_dir)
    plot_metric(df, "f1", "F1 Score", "f1_over_epochs", out_dir)
    plot_metric(df, "epoch_time", "Epoch Time", "epoch_time_over_epochs", out_dir)

    # time + Gantt
    plot_total_time(metrics, out_dir)
    plot_pipeline_gantt(metrics.get("pipeline_events", []), out_dir)
