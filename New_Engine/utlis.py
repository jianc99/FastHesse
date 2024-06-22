import torch
from FastHesse.New_Engine.model import Transformer
import numpy as np
import random

def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif ("cpu" in device) or ("mps" in device):
        pass
    else:
        print(f"device={device} is not yet suppported")

def load_model(checkpoint_path, device, precision, use_tp, rank_group=None, global_group = None):

    with torch.device('meta'):
        model = Transformer.from_name(checkpoint_path.parent.name)

    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    if "model" in checkpoint and "stories" in str(checkpoint_path):
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint, assign=True)

    if use_tp:
        from FastHesse.New_Engine.tp import apply_tp
        print("Applying tensor parallel to model ...")
        apply_tp(model, rank_group, global_group)

    model = model.to(device=device, dtype=precision)
    return model.eval()

def model_forward(model :Transformer, x :torch.Tensor, input_pos :torch.Tensor, cache_pos :torch.Tensor = None, attention_mask: torch.Tensor = None) -> torch.Tensor:
    return model(x, input_pos, cache_pos, attention_mask)

def model_forward_branch(model :Transformer, x :torch.Tensor, input_pos :torch.Tensor, cache_pos :torch.Tensor = None, attention_mask: torch.Tensor = None) -> torch.Tensor:
    return model(x, input_pos, cache_pos, attention_mask)

def prefill(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, cache_pos :torch.Tensor = None, attention_mask: torch.Tensor = None) -> torch.Tensor:
    # input_pos: [B, S]
    logits = model(x, input_pos, cache_pos, attention_mask)
    return logits

def gather_kv(model, indices, offsets):
    batch_size, indices_len = indices.size()
    storage_ids = offsets + torch.arange(0, indices_len, dtype=torch.long, device=indices.device).unsqueeze(0)
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=indices.device).view(batch_size,1).expand_as(indices)
    
    for b in model.layers:
        # Gather all k_cache and v_cache values at once
        new_k_cache = b.attention.kv_cache.k_cache[batch_indices, :, indices, :].permute(0,2,1,3)
        new_v_cache = b.attention.kv_cache.v_cache[batch_indices, :, indices, :].permute(0,2,1,3)        
        # Perform batch index_copy_
        for batch_idx in range(batch_size):
            b.attention.kv_cache.k_cache[batch_idx].index_copy_(
                dim=-2, index=storage_ids[batch_idx], source=new_k_cache[batch_idx]
            )
            b.attention.kv_cache.v_cache[batch_idx].index_copy_(
                dim=-2, index=storage_ids[batch_idx], source=new_v_cache[batch_idx]
            )

def cuda_graph_for_gather_kv(
                device="cuda:0", 
                batch_size=1, max_len=7, model=None, 
                n_warmups=3, mempool=None):
    
    static_offsets = torch.full((batch_size, 1), 1, dtype=torch.long, device=device)
    static_indices = torch.arange(0,max_len, dtype=torch.long, device=device).repeat(batch_size,1)
    
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmups):
            gather_kv(model, static_indices, static_offsets)
        s.synchronize()
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        gather_kv(model, static_indices, static_offsets)
    def run(indices, offsets):
        static_offsets.copy_(offsets)
        static_indices.copy_(indices)
        graph.replay()
    return run

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True