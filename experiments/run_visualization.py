# run_viz.py
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

    train_loss = metrics.get("train_loss", [])
    epochs = metrics.get("epochs", [])
    
    if len(epochs) != len(train_loss):
        epochs = list(range(1, len(train_loss) + 1))
    
    df = pd.DataFrame({
        "epoch": epochs,
        "loss": train_loss,
        "accuracy": metrics.get("accuracy", [float("nan")] * len(epochs)),
        "precision": metrics.get("precision", [float("nan")] * len(epochs)),
        "recall": metrics.get("recall", [float("nan")] * len(epochs)),
        "f1": metrics.get("f1", [float("nan")] * len(epochs)),
        "epoch_time": metrics.get("epoch_time", [float("nan")] * len(epochs))
    })

    return df, metrics


# -----------------------------------------------------------
# Metric plot
# -----------------------------------------------------------
def plot_metric(df, metric, ylabel, filename, out_dir):
    if metric not in df.columns or df[metric].isna().all():
        print(f"⚠️ No data for {metric}, skipping plot")
        return
    
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
# Training plots
# -----------------------------------------------------------
def generate_training_plots(file_path, mode):
    print(f"\n[Visualization] Generating training plots from {file_path}")
    
    df, metrics = load_metrics(file_path)
    out_dir = os.path.join("../results/plots", mode)
    os.makedirs(out_dir, exist_ok=True)
    
    plot_metric(df, "loss", "Training Loss", "loss_over_epochs", out_dir)
    plot_total_time(metrics, out_dir)
    
    print(f"[Visualization] Saved training plots to {out_dir}")


# -----------------------------------------------------------
# Inference plots
# -----------------------------------------------------------
def generate_inference_plots(file_path, mode):
    print(f"\n[Visualization] Generating inference plots from {file_path}")
    
    with open(file_path, "r") as f:
        metrics = json.load(f)
    
    out_dir = os.path.join("../results/plots", mode)
    os.makedirs(out_dir, exist_ok=True)
    
    metrics_to_plot = ["accuracy", "precision", "recall", "f1"]
    
    # Combined plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics_to_plot):
        if metric in metrics:
            ax = axes[idx]
            value = metrics[metric]
            ax.bar([metric.capitalize()], [value], color='skyblue', edgecolor='black')
            ax.set_title(f"{metric.capitalize()}: {value:.4f}")
            ax.set_ylabel("Score")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
    
    plt.suptitle(f"Inference Metrics (Total Time: {metrics.get('total_time', 0):.2f}s)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "inference_metrics.png"), dpi=300)
    plt.close()
    
    # Individual plots
    for metric in metrics_to_plot:
        if metric in metrics:
            plt.figure(figsize=(8, 6))
            plt.bar([metric.capitalize()], [metrics[metric]], color='lightcoral', width=0.6)
            plt.title(f"{metric.capitalize()}: {metrics[metric]:.4f}")
            plt.ylabel("Score")
            plt.ylim(0, 1)
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(out_dir, f"{metric}_bar.png"), dpi=300)
            plt.close()
    
    print(f"[Visualization] Saved inference plots to {out_dir}")


# -----------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------
print("=== GENERATING ALL VISUALIZATIONS ===\n")

# 1. Generate training plots
train_file = "../results/metrics/parallel_train_metrics.json"
if os.path.exists(train_file):
    print("1. Generating training plots...")
    generate_training_plots(train_file, mode="parallel")
else:
    print(f"❌ Training file not found: {train_file}")

# 2. Generate inference plots
infer_file = "../results/metrics/parallel_inference_metrics.json"
if os.path.exists(infer_file):
    print("\n2. Generating inference plots...")
    generate_inference_plots(infer_file, mode="parallel")
else:
    print(f"❌ Inference file not found: {infer_file}")

print("\n=== COMPLETE ===")
print("Check your plots in: ../results/plots/parallel/")