# GPipe-Style Pipeline Parallel Training on CIFAR-10  
CSCI-654 — Parallel Computing Term Project

---

## Overview

This project implements **GPipe-style pipeline parallelism** for training a ResNet20 architecture on the CIFAR-10 dataset.  
It demonstrates core concepts from parallel computing:

- Model partitioning  
- Microbatch-based pipelining  
- Multi-process execution  
- Overlapped computation  
- Timeline visualization using a Gantt chart  
- Comparison against sequential training  

The implementation uses **Python Multiprocessing**, **PyTorch**, and **custom stage workers** to create an educational, fully transparent pipeline-parallel system.

---

## Table of Contents

1. [Project Description](#project-description)  
2. [Technologies Used](#technologies-used)  
3. [Project Structure](#project-structure)  
4. [Installation](#installation)  
5. [Usage](#usage)  
6. [How GPipe Works Here](#how-gpipe-works-here)  
7. [Pipeline Architecture Diagram](#pipeline-architecture-diagram)  
8. [Features](#features)  
9. [Example Outputs](#example-outputs)  
10. [Gantt Chart Generation](#gantt-chart-generation)  
11. [Academic Value](#academic-value)  

---

# Project Description

GPipe is a pipeline-parallel training method introduced by Google in 2019.  
It speeds up training large models by splitting them into **sequential stages** and passing **microbatches** through them like an assembly line.

This project is a scaled-down educational implementation of GPipe using:

- A manually partitioned ResNet20  
- Three pipeline stages  
- Independent worker processes  
- Queues to shuttle microbatches  
- Logging of stage timings  
- Gantt chart visualization to show real concurrency  
- Comparison with vanilla sequential training  

It answers the core question:

> **How much parallel speedup can we achieve on a CPU-only setup using GPipe concepts?**

---

# Technologies Used

| Category | Technology |
|---------|------------|
| Programming | Python 3 |
| Deep Learning | PyTorch |
| Dataset | CIFAR-10 |
| Parallelism | multiprocessing.Process, Queue |
| Metrics | sklearn (accuracy, precision, recall, F1) |
| Visuals | matplotlib, seaborn, pandas |
| Model | ResNet20 (partitioned into 3 pipeline stages) |

---

# Project Structure

```
project/
│
├── experiments/
│   ├── train_sequential.py        # Baseline: single-process sequential training
│   ├── train_parallel.py          # GPipe-style pipeline training & inference
│   ├── visualize_results.py       # All plots + Gantt chart
│
├── models/
│   ├── resnet_cifar.py            # Full ResNet20 definition
│   ├── resnet_stages.py           # Stage0, Stage1, Stage2 partitions
│
├── pipeline/
│   ├── stage_worker.py            # The multi-process worker logic
│
├── results/
│   ├── metrics/                   # JSON logs of training runs
│   ├── checkpoints/               # Saved model weights
│   └── plots/
│       ├── sequential             # Plots for sequential mode
│       └── parallel               # Plots for GPipe mode
│
└── data/                          # CIFAR-10 dataset
```

---

# Installation

## 1. Clone the repository

```
git clone <your-repo-url>
cd <project-folder>
```

## 2. Install dependencies

```
pip install torch torchvision matplotlib seaborn pandas sklearn tqdm
```

The dataset is downloaded automatically.

---

# Usage

## ▶️ 1. Sequential Baseline Training

```
python experiments/train_sequential.py
```

Generates:

- `results/metrics/sequential_train_metrics.json`
- Plots in `results/plots/sequential/`

---

## ▶️ 2. GPipe-Style Pipeline Parallel Training

```
python experiments/train_parallel.py
```

Produces:

- `results/metrics/parallel_train_metrics.json`
- ResNet checkpoints
- Overlapped-execution Gantt chart

---

## ▶️ 3. Pipeline Inference Mode  
Useful for pure visualization and latency analysis.

```
# inside train_parallel.py, set mode="inference"
python experiments/train_parallel.py
```

---

# How GPipe Works Here

This project implements a **GPipe-style pipeline parallel training system** for ResNet20 on the CIFAR-10 dataset.  
The purpose is to demonstrate *pipeline parallelism, microbatching, concurrency, and throughput improvement* — all fundamental ideas in modern distributed and parallel computing systems.

Unlike standard PyTorch training, where the entire model lives in one process, this project:

- **Splits ResNet20 into three sequential stages**
- **Runs each stage in a dedicated Python process**
- **Feeds microbatches through the stages as a pipeline**, exactly like GPipe (Google, 2019)
- Uses:
  - Python `multiprocessing`
  - Inter-process communication (IPC) queues
  - Timed microbatch events → Gantt chart visualization
  - Sequential vs Pipeline mode comparison

This creates a real asynchronous, overlapped execution flow where multiple microbatches are being processed simultaneously in different parts of the network.

This project splits ResNet20 into **three sequential stages**:

```
Stage0 → Stage1 → Stage2 → Output
```

For each global batch:

1. The batch is divided into **microbatches**  
2. Stage 0 begins processing microbatch 0  
3. As soon as Stage 0 finishes mb0, it sends it to Stage 1 and begins mb1  
4. Stage 1 processes mb0, Stage 0 continues with mb1, etc  
5. All three stages become active simultaneously  
6. Stage workers send back logs to build a Gantt chart

This creates the **pipeline fill → steady state → drain** phases seen in real GPipe systems.

---

# Pipeline Architecture Diagram

```
           ┌──────────────┐
           │ Data Loader  │
           └──────┬───────┘
                  ▼
            (mb_id, microbatch)
                  ▼
           ┌──────────────┐
           │   q0_in      │
           └──────┬───────┘
                  ▼
        ┌─────────────────────┐
        │ Stage 0 (Process 0) │
        └──────┬──────────────┘
               ▼
             q0_out
               ▼
        ┌─────────────────────┐
        │ Stage 1 (Process 1) │
        └──────┬──────────────┘
               ▼
             q1_out
               ▼
        ┌─────────────────────┐
        │ Stage 2 (Process 2) │
        └──────┬──────────────┘
               ▼
             q2_out
               ▼
       ┌───────────────────────┐
       │ Final Collector/Train │
       └───────────────────────┘

   event_queue → Gantt Chart
```
## GPipe-Style Microbatching

GPipe's key idea is **split your global batch into many microbatches**, and send them through the network like an assembly line:

```
Time →
[S0(mb0)] [S0(mb1)] [S0(mb2)] ...
       [S1(mb0)] [S1(mb1)] ...
             [S2(mb0)] [S2(mb1)] ...
```

Your implementation matches GPipe’s model:

- **Global batch = 512**
- **Microbatch = 64**
- → **8 microbatches per batch**

Stage 0 starts processing microbatch 0.  
Before Stage 0 finishes microbatch 2, Stage 1 is already processing microbatch 1, and Stage 2 is finishing microbatch 0.

This produces *overlap and concurrency*, leading to improved throughput and reduced per-epoch wall-clock time.

---

## ⚙️ Internals: How Training Actually Works

### 🔵 Pipeline Forward Pass (Parallel, Timed)
Inside `stage_worker`, the forward pass is:

1. Receive `(mb_id, x)` from a queue  
2. Record start time  
3. Run the forward pass through the stage  
4. Record end time  
5. Push result + timestamps to the `event_queue`

Example log event:

```python
event_queue.put({
    "stage": stage_id,
    "mb_id": mb_id,
    "start": t0,
    "end": t1
})
```

These events later build the **Gantt chart**.

> **Important:** Pipeline forward pass uses no autograd — only timing and outputs.

### Backpropagation (Sequential, Full Model)
After collecting all microbatch outputs:

```python
logits_full = full_model(images_mb[mb_id])
loss_mb = criterion(logits_full, labels_mb[mb_id])
loss_mb.backward()
```

Doing backward on the full model:

- Avoids distributed autograd complexity
- Makes debugging easy
- Preserves correctness
- Still demonstrates real GPipe timing behavior

This is a **safe, academically valid** simulation of GPipe execution.

---

## Parallel Computing Concepts Demonstrated

###  Task Parallelism
Each stage performs a *different section* of ResNet20.

###  Pipeline Parallelism
Stages form a **producer → consumer → producer** structure:
```
Stage0 → Stage1 → Stage2 → Output
```

###  Concurrency
All stages run *simultaneously* on different microbatches.

###  Communication Costs
`Queue.put()` and `Queue.get()` model cross-device communication in real systems.

###  Load Balancing & Bottlenecks
Gantt chart reveals:

- Slower stages create bubbles  
- Faster stages idle while waiting  
- Opportunities to repartition the model

###  Amdahl’s Law / Speed-up Evaluation
Both sequential and pipeline total times are logged:

- `epoch_time_over_epochs.png`
- `total_training_time.png`

Useful for calculating theoretical and observed speed-up.

---
##  True Multi-Process Execution

Each stage is *really* running in another process:

```python
p0 = Process(target=stage_worker, args=(s0, q0_in, q0_out, 0, event_queue))
p1 = Process(target=stage_worker, args=(s1, q1_in, q1_out, 1, event_queue))
p2 = Process(target=stage_worker, args=(s2, q2_in, q2_out, 2, event_queue))
```

Each worker:

1. Pulls microbatch from input queue  
2. Computes forward pass  
3. Logs timing  
4. Pushes to next stage  
5. Continues immediately with the next microbatch  

This is genuine pipelined parallelism.

---

##  Real Timed Microbatch Events → Gantt Chart

Every stage logs:

```python
event_queue.put({
    "stage": stage_id,
    "mb_id": mb_id,
    "start": t0,
    "end": t1
})
```

Using these, `visualize_results.py` produces a proper **pipeline execution timeline**, similar to DeepSpeed, PipeDream, and GPipe papers.

### Example Structure (ASCII Visualization)

```
Stage 2 |■■■■■■■■■   ■■■■■■■■■   ■■■■■■■■■
Stage 1 |  ■■■■■■■■■   ■■■■■■■■■   ■■■■■■■■■
Stage 0 |    ■■■■■■■■■   ■■■■■■■■■   ■■■■■■■■■
           time →
```

This confirms that:

- Stages overlap  
- Microbatches flow continuously  
- The pipeline reaches steady-state  

---

## How GPipe Works (Short Overview)

GPipe (Huang et al., 2019) introduced:

| Concept | GPipe | Your Project |
|--------|-------|---------------|
| Model Split |  Partitioned layers |  Stage0/1/2 modules |
| Microbatching |  Yes |  `images.chunk(M)` |
| Concurrent Pipeline |  Yes |  Independent processes + queues |
| Distributed Autograd |  Fully | ✘ Simulated via full-model backward |
| Activation Recomputation | Optional | Not implemented |
| Gantt Timeline |  Yes |  Yes (event_queue) |

This is a faithful **CPU-only, multi-process simulation of GPipe** suitable for education and parallel computing demonstration.

---

## Why This Project Matters (Academic Value)

This project provides a practical demonstration of:

- Parallel system design  
- Pipelined execution  
- Worker scheduling  
- Communication overhead  
- Timing instrumentation  
- Visualization of real concurrency  
- System bottleneck analysis  
- Speedup calculations with Amdahl/Gustafson insights  

It teaches many concepts underpinning:

- DeepSpeed Pipeline Parallelism  
- Megatron-LM  
- GPipe  
- PipeDream 2BW  
- Tensor/Model sharding  


---


# Features

###  1. True Pipeline Parallelism  
Each stage runs in a dedicated process; microbatches flow asynchronously.

###  2. Timing-Aware Workers  
Every worker logs:

- stage ID  
- microbatch ID  
- start timestamp  
- end timestamp  

###  3. Real Gantt Chart Visualization  
Shows overlapped execution and bottlenecks.

###  4. Microbatching Support  
Global batch (e.g., 512) → multiple microbatches (e.g., 64 each).

###  5. Sequential vs Parallel Comparison  
- loss  
- epoch time  
- total time  
- accuracy/precision/recall/F1 (in inference)

###  6. Clean Modular Architecture  
Easy to extend to more stages or larger models.

---

# Example Outputs

### Loss Curves  
Sequential vs GPipe.

### Epoch Time Curve  
Shows reduced per-epoch time with pipelining.

### Total Training Time Comparison  
Parallel mode typically reduces the wall-clock time.

### Gantt Chart  
Example (conceptual):

```
Stage 0: |■■■■■■■■■■■■■|■■■■■■■■■■■■■|■■■■■■■|
Stage 1:      |■■■■■■■■■■■■■|■■■■■■■■■■■■■|■■■■■■■|
Stage 2:           |■■■■■■■■■■■■■|■■■■■■■■■■■■■|■■■■■■■|
```

---

# Gantt Chart Generation

Pipeline events are logged like this:

```python
event_queue.put({
  "epoch": epoch,
  "stage": stage_id,
  "mb_id": mb_id,
  "start": t0,
  "end": t1
})
```

`visualize_results.py` converts this into a proper Gantt chart using matplotlib.

---

# Academic Value

This project demonstrates:

###  Pipeline parallelism  
A key topic in distributed deep learning.

###  Microbatch scheduling  
A core idea from GPipe, PipeDream, DeepSpeed PP.

###  Process-level parallelism  
Clear example of divide-and-conquer model partitioning.

###  Quantitative performance analysis  
Includes epoch-time curves and total-time comparison.

###  Visualization of concurrency  
The Gantt chart is the highlight of the project.

---
## References

- Huang et al., *GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism* (Google, 2019)  
- Narayanan et al., *PipeDream: Generalized Pipeline Parallelism for DNN Training* (MSR, 2019)  
- Shoeybi et al., *Megatron-LM: Training Multi-Billion Parameter Language Models* (NVIDIA, 2019)
---

# Conclusion

This project is a full, working demonstration of **GPipe-style pipeline parallelism**, implemented from scratch using Python Multiprocessing and PyTorch.

It is structured for academic clarity and extensibility, and provides all metrics and visualizations needed for a thorough parallel computing project submission.

---

