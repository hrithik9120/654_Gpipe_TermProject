import sys
import os
import time
import json
import multiprocessing as mp
import torch
from multiprocessing import Process, Queue
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) 

from models.resnet_cifar import ResNet20
from models.resnet_stages import Stage0, Stage1, Stage2, Stage3, Stage4, Stage5
from pipeline.stage_worker import stage_worker
from experiments import visualize_results

NUM_STAGES = 6 
device = "cpu"

def run_pipeline_speedtest(
    microbatch_size=64,
    global_batch_size=512,
    num_epochs=20,
    checkpoint_path="../results/checkpoints/resnet20_epoch_5.pth"
):

    assert global_batch_size % microbatch_size == 0 # Ensure divisibility of batches
    M = global_batch_size // microbatch_size # Number of microbatches

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),
    ])

    trainset = datasets.CIFAR10("../data", train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=global_batch_size,
                             shuffle=True, drop_last=True)

    full_model = ResNet20()
    if os.path.exists(checkpoint_path):
        print(f"Loading initial weights from {checkpoint_path}")
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

    # Create queues and stage workers
    queues = [Queue(8) for _ in range(NUM_STAGES + 1)]
    event_queue = Queue()

    workers = []
    for i in range(NUM_STAGES):
        p = Process(
            target=stage_worker,
            args=(stages[i], queues[i], queues[i+1], i, True, event_queue)
        )
        p.start()
        workers.append(p)

    final_queue = queues[-1]
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(full_model.parameters(), lr=0.01, momentum=0.9)

    print("\nStarting GPipe speed-focused training...\n")

    total_start = time.time()

    metrics = {
        "epochs": list(range(1, num_epochs + 1)),
        "train_loss": [],
        "epoch_time": [],
        "pipeline_events": [],
        "total_time": None
    }

    
    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_start = time.time()

        print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")

        for images, labels in trainloader:

            optimizer.zero_grad()
            images_mb = images.chunk(M)
            labels_mb = labels.chunk(M)
            batch_loss = 0.0

            # Send microbatches to pipeline
            for mb_id in range(M):
                queues[0].put((mb_id, images_mb[mb_id]))

            received = 0 # number of microbatches received
            outputs = [] # Collect outputs from the final stage

            # Collect outputs from the final stage
            while received < M:
                msg = final_queue.get()
                if msg is None:
                    continue
                mb_id, out = msg[0], msg[1]
                outputs.append((mb_id, out))
                received += 1

            # Sort outputs by microbatch ID
            outputs.sort(key=lambda x: x[0])

            # Compute grads using full model
            for mb_id, (_, out) in enumerate(outputs):
                logits = full_model(images_mb[mb_id])
                loss = criterion(logits, labels_mb[mb_id]) / M
                loss.backward()
                batch_loss += loss.item()

            optimizer.step()
            epoch_loss += batch_loss

        epoch_time = time.time() - epoch_start
        metrics["train_loss"].append(epoch_loss)
        metrics["epoch_time"].append(epoch_time)

        print(f"Epoch {epoch+1} Loss = {epoch_loss:.4f}")
        print(f"Epoch {epoch+1} Time = {epoch_time:.2f} sec")

    print("\nShutting down pipeline...")

    # Send termination to workers
    for q in queues[:-1]:
        q.put(None)

    time.sleep(0.5) 

    for p in workers:
        p.terminate()
        p.join()

    # Drain pipeline events
    while not event_queue.empty():
        metrics["pipeline_events"].append(event_queue.get_nowait())

    metrics["total_time"] = time.time() - total_start
    print(f"\nTotal Time = {metrics['total_time']:.2f} sec")

    # Save metric file
    os.makedirs("../results/metrics", exist_ok=True)
    out_file = "../results/metrics/parallel_speedtest_metrics.json"
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=4)

    # Use training plot generator
    visualize_results.generate_training_plots(out_file, mode="parallel")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    run_pipeline_speedtest()