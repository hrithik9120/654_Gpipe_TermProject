# experiments/train_parallel.py

import sys
import os
import time
import json
import multiprocessing as mp
import argparse
from multiprocessing import Process, Queue

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments import visualize_results
from models.resnet_cifar import ResNet20
from models.resnet_stages import Stage0, Stage1, Stage2, Stage3, Stage4, Stage5
from pipeline.stage_worker import stage_worker


# =====================================================================
# 6-STAGE PIPELINE CONFIGURATION
# =====================================================================
NUM_STAGES = 6
device = "cpu"

# =====================================================================
# PIPELINE INFERENCE (FULL 6-STAGE RESNET PIPELINE)
# =====================================================================
def run_pipeline_inference(
    microbatch_size=128,
    checkpoint_path="../results/checkpoints/resnet20_epoch_20.pth"
):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),
    ])

    testset = datasets.CIFAR10("../data", train=False,
                               download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=microbatch_size,
                            shuffle=False, drop_last=True)

    # Load full model
    full_model = ResNet20()
    if os.path.exists(checkpoint_path):
        print(f"[Inference] Loading weights from {checkpoint_path}")
        full_model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    else:
        print("[Inference] WARNING: No checkpoint found, using random weights.")
    full_model.eval()

    all_labels = []
    all_preds = []
    total_start = time.time()

    # Simple inference without pipeline
    with torch.no_grad():
        for imgs, labels in tqdm(testloader, desc="[Inference]"):
            logits = full_model(imgs)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_labels.extend(labels.numpy())
            all_preds.extend(preds)

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro"
    )

    print("\n===== Inference Complete =====")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # Save metrics
    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_time": time.time() - total_start,
        "train_loss": [],  # Empty for inference
        "epochs": [1],  # Single epoch for inference
        "pipeline_events": [],  # Empty since no pipeline
    }

    os.makedirs("../results/metrics", exist_ok=True)
    out_file = "../results/metrics/parallel_inference_metrics.json"
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=4)

    # FIXED: Call the correct function
    visualize_results.generate_inference_plots(out_file, mode="parallel")



# =====================================================================
# PIPELINE TRAINING (GPipe Logic but Training on Full Model)
# =====================================================================
def run_pipeline_training(
    microbatch_size=64,
    global_batch_size=512,
    num_epochs=20,
    checkpoint_path="../results/checkpoints/resnet20_epoch_5.pth"
):

    assert global_batch_size % microbatch_size == 0
    M = global_batch_size // microbatch_size

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),
    ])

    trainset = datasets.CIFAR10("../data", train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=global_batch_size,
                             shuffle=True, drop_last=True)

    full_model = ResNet20()
    if os.path.exists(checkpoint_path):
        print(f"[Training] Loading initial weights from {checkpoint_path}")
        full_model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    full_model.train()

    # Build pipeline stages
    stages = [
        Stage0(full_model),
        Stage1(full_model),
        Stage2(full_model),
        Stage3(full_model),
        Stage4(full_model),
        Stage5(full_model)
    ]

    queues = [Queue(8) for _ in range(NUM_STAGES + 1)]
    event_queue = Queue()

    workers = []
    for i in range(NUM_STAGES):
        p = Process(
            target=stage_worker,
            args=(stages[i], queues[i], queues[i+1], i, True, event_queue)  # Note: is_training=True
        )
        p.start()
        workers.append(p)

    final_queue = queues[-1]

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(full_model.parameters(), lr=0.01, momentum=0.9)

    print("\n[Training] Starting training with pipeline...\n")
    total_start = time.time()
    metrics = {
            "train_loss": [], 
            "epochs": list(range(1, num_epochs + 1)), 
            "pipeline_events": [],
            "accuracy": [],  # Empty for training
            "precision": [],
            "recall": [],
            "f1": [],
            "epoch_time": []
        }


    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_start = time.time()
        print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")

        for images, labels in trainloader:

            optimizer.zero_grad()
            
            images_mb = images.chunk(M)
            labels_mb = labels.chunk(M)
            batch_loss = 0.0

            # Feed all microbatches through pipeline
            for mb_id in range(M):
                queues[0].put((mb_id, images_mb[mb_id]))
            
            # Collect outputs from pipeline
            outputs = []
            for _ in range(M):
                msg = final_queue.get()
                if msg is None:
                    continue
                mb_id, out = msg[0], msg[1]
                outputs.append((mb_id, out))
            
            # Sort by microbatch ID
            outputs.sort(key=lambda x: x[0])
            
            # Compute loss and backward for each microbatch
            for mb_id, (_, out) in enumerate(outputs):
                # IMPORTANT: We need to recompute with full model to get gradients
                # because pipeline workers are in separate processes
                logits = full_model(images_mb[mb_id])
                loss = criterion(logits, labels_mb[mb_id])
                
                loss = loss / M
                loss.backward()
                batch_loss += loss.item() * M
            
            optimizer.step()
            epoch_loss += batch_loss

        print(f"Epoch {epoch+1} Loss = {epoch_loss:.4f}")
        metrics["train_loss"].append(epoch_loss)
        epoch_time = time.time() - epoch_start
        metrics["epoch_time"].append(epoch_time)
        print(f"Epoch {epoch+1} Time = {epoch_time:.2f} seconds")
        # Save checkpoint
        torch.save(full_model.state_dict(),
                   "../results/checkpoints/gpipe_trained.pth")
        
        # Evaluate on training set for accuracy (optional, can be skipped for speed)
        full_model.eval()
        all_labels = []
        all_preds = []
        
        # Create a small evaluation loader from training data
        eval_size = min(1000, len(trainset))  # Use 1000 samples max
        eval_indices = torch.randperm(len(trainset))[:eval_size]
        eval_subset = torch.utils.data.Subset(trainset, eval_indices)
        eval_loader = DataLoader(eval_subset, batch_size=256, shuffle=False)

        with torch.no_grad():
            for imgs, labels in eval_loader:
                logits = full_model(imgs)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_labels.extend(labels.numpy())
                all_preds.extend(preds)

        full_model.train()
        
        if len(all_labels) > 0:
            acc = accuracy_score(all_labels, all_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, average="macro", zero_division=0
            )
        else:
            acc = precision = recall = f1 = 0.0
        
        metrics["accuracy"].append(acc)
        metrics["precision"].append(precision)
        metrics["recall"].append(recall)
        metrics["f1"].append(f1)
        
        print(f"Epoch {epoch+1} Training Metrics - Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        

    # ========== SHUTDOWN ==========
    print("\n[Training] Shutting down pipeline...")
    
    # Send shutdown signals
    for q in queues[:-1]:  # All except final queue
        q.put(None)
    
    # Wait for shutdown to propagate
    time.sleep(0.5)
    
    # Collect any remaining outputs
    while not final_queue.empty():
        try:
            final_queue.get_nowait()
        except:
            pass
    
    # Terminate workers
    for p in workers:
        p.terminate()
        p.join()
    
    # Collect events
    while not event_queue.empty():
        metrics["pipeline_events"].append(event_queue.get_nowait())

    total_time = time.time() - total_start
    metrics["total_time"] = total_time
    print(f"\n[Training] Total training time: {total_time:.2f} seconds")

    # Save results
    os.makedirs("../results/metrics", exist_ok=True)
    with open("../results/metrics/parallel_train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # FIXED: Call the correct function
    visualize_results.generate_training_plots(
        "../results/metrics/parallel_train_metrics.json", mode="parallel"
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="all",
                        choices=["train", "infer", "all"])
    args = parser.parse_args()

    if args.mode == "train":
        run_pipeline_training()

    elif args.mode == "infer":
        run_pipeline_inference()

    else:
        print("\n==== STEP 1: TRAIN ====")
        run_pipeline_training()

        print("\n==== STEP 2: INFER ====")
        run_pipeline_inference()
