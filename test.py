import torch
import torch.distributed as dist
import os

def _get_global_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))

def _get_world_size() -> int:
    return int(os.environ.get("LOCAL_WORLD_SIZE", "1"))

def init_dist():
    global_rank = _get_global_rank()
    world_size = _get_world_size()
    torch.cuda.set_device(global_rank)
    dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size)
    draft_group = dist.new_group([0,1,2])
    target_group = dist.new_group([3,4])
    return draft_group, target_group

if __name__ == "__main__":
    init_dist()