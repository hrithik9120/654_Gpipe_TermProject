import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torch.utils.data import DataLoader
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.resnet_cifar import ResNet20
from tqdm import tqdm
import os, json, time
import visualize_results

device = "cpu"


def evaluate(model, dataloader):
    """Evaluate model on validation/test set."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)

    return acc, precision, recall, f1, cm


def train_sequential(epochs=20, batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    trainset = datasets.CIFAR10(root="../data", train=True, download=True, transform=transform)
    testset  = datasets.CIFAR10(root="../data", train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader  = DataLoader(testset, batch_size=256)

    # -------------------------
    # MODEL + LOSS + OPTIMIZER
    # -------------------------
    model = ResNet20().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # -------------------------
    # METRICS STORAGE
    # -------------------------
    metrics = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "epoch_time": [],
        "confusion_matrix": None,
    }

    os.makedirs("../results/metrics", exist_ok=True)
    os.makedirs("../results/checkpoints", exist_ok=True)

    total_start = time.time()

    # -------------------------
    # TRAINING LOOP
    # -------------------------
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0

        tqdm_loader = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")

        for images, labels in tqdm_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            tqdm_loader.set_postfix({"loss": loss.item()})

        # -------------------------
        # EVALUATE
        acc, precision, recall, f1, cm = evaluate(model, testloader)
        epoch_time = time.time() - epoch_start

        metrics["epochs"].append(epoch + 1)
        metrics["train_loss"].append(running_loss / len(trainloader))
        # store "validation" loss as 1-accuracy proxy if exact loss isn't needed
        metrics["val_loss"].append(1.0 - acc)
        metrics["accuracy"].append(acc)
        metrics["precision"].append(precision)
        metrics["recall"].append(recall)
        metrics["f1"].append(f1)
        metrics["epoch_time"].append(epoch_time)
        metrics["confusion_matrix"] = cm.tolist()

        # Save checkpoint
        torch.save(model.state_dict(), f"../results/checkpoints/resnet20_epoch_{epoch+1}.pth")

        print(f"\nEpoch {epoch+1} Results:")
        print(f"Loss: {running_loss/len(trainloader):.4f}")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"Epoch Time: {epoch_time:.2f}s")
        print("-" * 60)

    total_time = time.time() - total_start
    metrics["total_training_time"] = total_time

    # Save metrics JSON
    metrics_path = "../results/metrics/sequential_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # Generate plots for sequential run
    visualize_results.generate_all_plots(
        metrics_path,
        mode="sequential"
    )
    print("\nSequential Training Complete")
    print("Total Training Time:", total_time)


if __name__ == "__main__":
    train_sequential()
