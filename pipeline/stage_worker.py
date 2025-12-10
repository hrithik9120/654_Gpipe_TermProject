import time
import torch

def stage_worker(stage_module, in_queue, out_queue, stage_id, is_training, event_queue):
    """
    A single stage worker in the pipeline. Picks up microbatches from input queue and performs forward pass. Populates
    output queue after forward pass.
    """
    torch.set_num_threads(1)
    print(f"[Stage {stage_id}] Worker started")
    
    # Use a reasonable timeout
    TIMEOUT = 30.0

    while True:
        try:
            item = in_queue.get(timeout=TIMEOUT)
        except:
            print(f"[Stage {stage_id}] In-queue timeout after {TIMEOUT} seconds, shutting down.")
            if out_queue is not None:
                out_queue.put(None)
            break

        if item is None:
            if out_queue is not None:
                out_queue.put(None)
            break

        if len(item) == 2:
            mb_id, x = item
        elif len(item) == 5:
            mb_id, x, _, _, _ = item
        else:
            print(f"[Stage {stage_id}] Unexpected item format: {item}")
            continue

        t0 = time.time()

        # Compute forward pass
        with torch.set_grad_enabled(is_training):
            out = stage_module(x)

        t1 = time.time()

        event_queue.put({
            "stage": stage_id,
            "mb_id": mb_id,
            "start": t0,
            "end": t1,
            "queue_wait": 0.0
        })

        if out_queue is not None:
            # Detach the output before sending through queue
            out_queue.put((mb_id, out.detach(), t0, t1, stage_id))