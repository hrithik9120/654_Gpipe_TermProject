# pipeline/stage_worker.py
import time
import torch


def stage_worker(stage_module, in_queue, out_queue, stage_id, training=False,
                 event_queue=None):
    """
    Generic stage worker.
    - Reads (mb_id, x) from in_queue
    - Runs stage_module(x)
    - Writes (mb_id, y, t0, t1, stage_id) to out_queue
    - Optionally logs timing events to event_queue for Gantt chart
    """
    torch.set_num_threads(1)
    stage_module.eval()  # we don't train inside the pipeline

    print(f"[Stage {stage_id}] Worker started", flush=True)

    while True:
        item = in_queue.get()
        if item is None:
            # propagate shutdown to the next stage and exit
            if out_queue is not None:
                out_queue.put(None)
            print(f"[Stage {stage_id}] Worker exiting.", flush=True)
            break

        mb_id, x = item  # ALWAYS 2-tuple from main process

        t0 = time.time()
        with torch.no_grad():
            y = stage_module(x)
        t1 = time.time()

        # send downstream (consistent 5-tuple)
        if out_queue is not None:
            out_queue.put((mb_id, y, t0, t1, stage_id))

        # log event for Gantt
        if event_queue is not None:
            event_queue.put({
                "stage": int(stage_id),
                "mb_id": int(mb_id),
                "start": float(t0),
                "end": float(t1),
            })
