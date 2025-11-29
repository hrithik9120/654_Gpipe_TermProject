import torch

def split_microbatches(batch, M):
    return batch.chunk(M)

def gather_microbatch_outputs(microbatch_outputs):
    return torch.cat(microbatch_outputs, dim=0)