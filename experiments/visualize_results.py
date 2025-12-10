import json
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set(style="whitegrid")

def load_metrics(file_path):
    """
    Load metrics from JSON file into DataFrame
    """
    with open(file_path, "r") as f:
        metrics = json.load(f)

    # Get train_loss and ensure epochs array matches its length
    train_loss = metrics.get("train_loss", [])
    epochs = metrics.get("epochs", [])
    
    # If epochs not provided or wrong length, create it
    if len(epochs) != len(train_loss):
        epochs = list(range(1, len(train_loss) + 1))
    
    # Create DataFrame with consistent lengths
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

def plot_metric(df, metric, ylabel, filename, out_dir):
    """
    Plot a single training metric over epochs
    """
    # Check if we have valid data for this metric
    if metric not in df.columns or df[metric].isna().all():
        print(f"No data for {metric}, skipping plot")
        return
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="epoch", y=metric, marker="o")
    plt.title(f"{ylabel} Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.savefig(os.path.join(out_dir, f"{filename}.png"), dpi=300)
    plt.close()

def plot_total_time(metrics, out_dir, filename="total_training_time"):
    """
    Plot total training time as a bar chart
    """
    total_time = metrics.get("total_training_time") or metrics.get("total_time")
    if total_time is None:
        print("No total training time found.")
        return

    plt.figure(figsize=(6, 6))
    sns.barplot(x=["Model"], y=[total_time])
    plt.title("Total Training Time")
    plt.ylabel("Time (seconds)")
    plt.savefig(os.path.join(out_dir, f"{filename}.png"), dpi=300)
    plt.close()

def plot_confusion_matrix(cm, out_dir, filename="confusion_matrix"):
    """
    Plot confusion matrix heatmap
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(out_dir, f"{filename}.png"), dpi=300)
    plt.close()

def plot_butterfly(events, out_dir):
    """
    Create a butterfly plot to visualize load imbalance
    """
    if not events:
        return
    
    df = pd.DataFrame(events)
    df["duration"] = df["end"] - df["start"]
    df["start_rel"] = df["start"] - df["start"].min()

    plt.figure(figsize=(12,6))
    plt.scatter(df["start_rel"], df["stage"], s=df["duration"]*500, alpha=0.5)
    plt.xlabel("Time (relative)")
    plt.ylabel("Stage")
    plt.title("Pipeline Butterfly Plot (Load Imbalance Visualization)")
    plt.savefig(os.path.join(out_dir, "butterfly_plot.png"))
    plt.close()

def plot_pipeline_gantt(events, out_dir, filename="pipeline_gantt"):
    """
    Create a Gantt chart for pipeline stages
    """
    if not events:
        print("Gantt chart: no pipeline events")
        return

    df = pd.DataFrame(events)

    # ensure required columns
    if not {"stage", "start", "end"}.issubset(df.columns):
        print("Gantt chart: events missing stage/start/end")
        return

    # Compute duration and relative start time
    df["duration"] = df["end"] - df["start"]
    df["start_rel"] = df["start"] - df["start"].min()

    # Gantt chart
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

    # Gantt chart formatting
    plt.xlabel("Time (sec, relative)")
    plt.ylabel("Pipeline Stage")
    plt.title("Pipeline Gantt Chart")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{filename}.png"), dpi=300)
    plt.close()

    # Stage latency bar plot
    plt.figure(figsize=(8,5))
    sns.barplot(x="stage", y="duration", data=df)
    plt.xlabel("Stage ID")
    plt.ylabel("Avg Duration (s)")
    plt.title("Average Stage Latency")
    plt.savefig(os.path.join(out_dir, "stage_latency.png"))
    plt.close()

def plot_training_metrics_grid(df, out_dir):
    """
    Creates a grid of all the training metrics to see them side-by-side.
    """
    metrics_to_plot = ["loss", "accuracy", "precision", "recall", "f1", "epoch_time"]
    titles = ["Training Loss", "Accuracy", "Precision", "Recall", "F1 Score", "Epoch Time (s)"]
    
    _, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
        ax = axes[i]
        if metric in df.columns and not df[metric].isna().all():
            ax.plot(df["epoch"], df[metric], marker='o', linewidth=2, markersize=6)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(title.split("")[-1])
            ax.grid(True, alpha=0.3)
            
            if len(df[metric]) > 0 and not pd.isna(df[metric].iloc[-1]):
                last_val = df[metric].iloc[-1]
                ax.annotate(f'{last_val:.4f}', 
                           xy=(df["epoch"].iloc[-1], last_val),
                           xytext=(10, 0), textcoords='offset points',
                           fontsize=9, color='red')
        else:
            ax.text(0.5, 0.5, f'No {metric} data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
    
    plt.suptitle("Training Metrics Over Epochs", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_metrics_grid.png"), dpi=300)
    plt.close()

def generate_training_plots(file_path, mode):
    """
    Generates plots for all training metrics.
    """
    print(f"\nGenerating training plots from {file_path}")
    
    df, metrics = load_metrics(file_path)
    out_dir = os.path.join("../results/plots", mode)
    os.makedirs(out_dir, exist_ok=True)
    
    plot_metric(df, "loss", "Training Loss", "loss_over_epochs", out_dir)
    plot_metric(df, "accuracy", "Accuracy", "accuracy_over_epochs", out_dir)
    plot_metric(df, "precision", "Precision", "precision_over_epochs", out_dir)
    plot_metric(df, "recall", "Recall", "recall_over_epochs", out_dir)
    plot_metric(df, "f1", "F1 Score", "f1_over_epochs", out_dir)
    plot_metric(df, "epoch_time", "Epoch Time", "epoch_time_over_epochs", out_dir)
    
    plot_training_metrics_grid(df, out_dir)
    plot_total_time(metrics, out_dir)
    
    if mode == "parallel":
        events = metrics.get("pipeline_events", [])
        if events and len(events) > 0:
            print(f"Found {len(events)} pipeline events")
            plot_pipeline_gantt(events, out_dir)
            plot_butterfly(events, out_dir)
        else:
            print("No pipeline events found to create charts.")
    
    print(f"Saved training plots to {out_dir}")


def generate_inference_plots(file_path, mode):
    """
    Generates plots for all inference metrics.
    """
    print(f"\nGenerating inference plots from {file_path}")
    
    with open(file_path, "r") as f:
        metrics = json.load(f)
    
    out_dir = os.path.join("../results/plots", mode)
    os.makedirs(out_dir, exist_ok=True)
    
    metrics_to_plot = ["accuracy", "precision", "recall", "f1"]

    plt.figure(figsize=(12, 8))
    
    _, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics_to_plot):
        if metric in metrics:
            ax = axes[i]
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
    
    print(f"Saved inference plots to {out_dir}")


def generate_all_plots(file_path, mode):
    """
    Main entry point for generating all plots.
    """
    print(f"\nGenerating plots for the file path: {file_path}")
    
    with open(file_path, "r") as f:
        metrics = json.load(f)
    
    if "train_loss"in metrics and len(metrics["train_loss"]) > 0:
        generate_training_plots(file_path, mode)
    elif "accuracy"in metrics:
        generate_inference_plots(file_path, mode)
    else:
        print(f"Could not load file: {file_path}")