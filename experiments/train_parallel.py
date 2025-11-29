# experiments/train_parallel.py

import sys
import os
import time
import json
import multiprocessing as mp
from multiprocessing import Process, Queue

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments import visualize_results
from models.resnet_cifar import ResNet20
from models.resnet_stages import Stage0, Stage1, Stage2
from pipeline.stage_worker import stage_worker


device = "cpu"


# -------------------------------------------------------------------
# Inference pipeline (for metrics + pipeline Gantt on test set)
# -------------------------------------------------------------------
def run_pipeline_inference(
    microbatch_size=128,
    checkpoint_path="../results/checkpoints/resnet20_epoch_5.pth"
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

    # Full model
    full_model = ResNet20()
    if os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}")
        full_model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    full_model.eval()

    # Stage modules
    s0 = Stage0(full_model)
    s1 = Stage1(full_model)
    s2 = Stage2(full_model)

    # Queues
    q0 = Queue(maxsize=8)
    q1 = Queue(maxsize=8)
    q2 = Queue(maxsize=8)
    q3 = Queue(maxsize=8)
    event_queue = Queue()

    # Workers
    p0 = Process(target=stage_worker, args=(s0, q0, q1, 0, False, event_queue))
    p1 = Process(target=stage_worker, args=(s1, q1, q2, 1, False, event_queue))
    p2 = Process(target=stage_worker, args=(s2, q2, q3, 2, False, event_queue))

    p0.start(); p1.start(); p2.start()

    metrics = {
        "epochs": [1],   # single "epoch" of inference
        "train_loss": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "epoch_time": [],
        "pipeline_events": [],
        "total_time": None,
    }

    label_store = {}
    pred_store = []

    total_start = time.time()
    epoch_start = time.time()

    # feed microbatches
    mb_id = 0
    for imgs, labels in tqdm(testloader, desc="[Pipeline Inference - feeding]"):
        label_store[mb_id] = labels.numpy()
        q0.put((mb_id, imgs))
        mb_id += 1

    num_microbatches = mb_id

    # collect outputs from final stage
    from queue import Empty
    received = set()
    timeout = 10

    while len(received) < num_microbatches:
        try:
            msg = q3.get(timeout=timeout)
        except Empty:
            print("\n[Main] Timeout waiting for Stage 2.")
            break

        if msg is None:
            continue

        mb_out, logits, t0, t1, stage = msg  # stage should be 2
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        pred_store.append((mb_out, preds))
        received.add(mb_out)

    # shut down workers
    q0.put(None)
    p0.join(); p1.join(); p2.join()

    # collect timing events for Gantt
    pipeline_events = []
    from queue import Empty as QEmpty
    while True:
        try:
            ev = event_queue.get_nowait()
        except QEmpty:
            break
        pipeline_events.append(ev)

    # sort predictions
    pred_store.sort(key=lambda x: x[0])

    all_labels = []
    all_preds = []

    for i in range(len(pred_store)):
        all_labels.extend(label_store[i])
        all_preds.extend(pred_store[i][1])

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro"
    )

    metrics["accuracy"].append(acc)
    metrics["precision"].append(precision)
    metrics["recall"].append(recall)
    metrics["f1"].append(f1)
    metrics["epoch_time"].append(time.time() - epoch_start)
    metrics["total_time"] = time.time() - total_start
    metrics["pipeline_events"] = pipeline_events

    print("\n===== Pipeline Inference Complete =====")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")

    os.makedirs("../results/metrics", exist_ok=True)
    with open("../results/metrics/parallel_inference_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    visualize_results.generate_all_plots(
        "../results/metrics/parallel_inference_metrics.json",
        mode="parallel"
    )


# -------------------------------------------------------------------
# GPipe-style parallel TRAINING (no metrics except loss + time)
# -------------------------------------------------------------------
def run_pipeline_training(
    microbatch_size=64,
    global_batch_size=512,
    num_epochs=8,
    checkpoint_path="../results/checkpoints/resnet20_epoch_5.pth"
):

    assert global_batch_size % microbatch_size == 0
    M = global_batch_size // microbatch_size

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),
    ])

    trainset = datasets.CIFAR10("../data", train=True,
                                download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=global_batch_size,
                             shuffle=True, drop_last=True)

    # full model
    full_model = ResNet20()
    if os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}")
        full_model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    full_model.train()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(full_model.parameters(), lr=0.01, momentum=0.9)

    # Stage modules share weights with full_model
    s0 = Stage0(full_model)
    s1 = Stage1(full_model)
    s2 = Stage2(full_model)

    # queues for pipeline
    q0 = Queue(maxsize=8)
    q1 = Queue(maxsize=8)
    q2 = Queue(maxsize=8)
    q3 = Queue(maxsize=8)
    event_queue = Queue()

    # launch workers (no autograd, just forward timing)
    p0 = Process(target=stage_worker, args=(s0, q0, q1, 0, False, event_queue))
    p1 = Process(target=stage_worker, args=(s1, q1, q2, 1, False, event_queue))
    p2 = Process(target=stage_worker, args=(s2, q2, q3, 2, False, event_queue))

    p0.start(); p1.start(); p2.start()

    print("\n[Training] Starting GPipe-style multi-epoch training...\n")

    metrics = {
        "epochs": list(range(1, num_epochs + 1)),
        "train_loss": [],
        "epoch_time": [],
        "pipeline_events": [],
        "total_time": None
    }

    total_start = time.time()
    pipeline_events = []

    for epoch in range(num_epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")

        for images, labels in trainloader:
            optimizer.zero_grad()

            images_mb = images.chunk(M)
            labels_mb = labels.chunk(M)
            num_microbatches = len(images_mb)

            # 1. pipeline forward (timed by workers)
            for mb_id in range(num_microbatches):
                q0.put((mb_id, images_mb[mb_id]))

            # 2. collect outputs just to keep pipeline flowing
            finished = 0
            while finished < num_microbatches:
                msg = q3.get()
                if msg is None:
                    continue
                mb_id, logits_stage2, t0, t1, stage = msg
                # we don't use logits_stage2 for loss; we just want pipeline behaviour
                finished += 1

            # 3. REAL TRAINING — full forward + backward
            for mb_id in range(num_microbatches):
                logits_full = full_model(images_mb[mb_id])
                loss_mb = criterion(logits_full, labels_mb[mb_id])
                loss_mb.backward()
                epoch_loss += loss_mb.item()

            optimizer.step()

        epoch_time = time.time() - epoch_start
        metrics["train_loss"].append(float(epoch_loss))
        metrics["epoch_time"].append(epoch_time)

        torch.save(full_model.state_dict(), "../results/checkpoints/gpipe_trained.pth")
        print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")

    metrics["total_time"] = time.time() - total_start

    # stop workers
    q0.put(None)
    p0.join(); p1.join(); p2.join()

    # drain event queue into pipeline_events
    from queue import Empty as QEmpty
    while True:
        try:
            ev = event_queue.get_nowait()
        except QEmpty:
            break
        pipeline_events.append(ev)

    metrics["pipeline_events"] = pipeline_events

    print(f"\nTotal Training Time: {metrics['total_time']:.2f} seconds")

    os.makedirs("../results/metrics", exist_ok=True)
    with open("../results/metrics/parallel_train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    visualize_results.generate_all_plots(
        "../results/metrics/parallel_train_metrics.json",
        mode="parallel"
    )

    print("\nGPipe-style training complete.\n")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    mode = "g_train"   # "inference" OR "g_train"

    if mode == "inference":
        run_pipeline_inference()
    elif mode == "g_train":
        run_pipeline_training()
