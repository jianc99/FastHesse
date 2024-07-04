import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from FastHesse.Engine.model import Transformer
from torch.nn.attention import SDPBackend, sdpa_kernel
from FastHesse.Engine.utils import load_model, cuda_graph_for_gather_kv

class LMBackend:
    def __init__(self, dtype = torch.bfloat16, device: str = "cuda:0", dec_list: list = [1]) -> None:
        self.dtype = dtype
        self.device = device
        self.model_forward = {}
        for dec_len in dec_list:
            if dec_len == 0: continue
            self.model_forward[dec_len] = lambda model, x, input_pos, cache_pos, attention_mask : model(x, input_pos, cache_pos, attention_mask)
        self.prefill = lambda model, x, input_pos, cache_pos, attention_mask : model(x, input_pos, cache_pos, attention_mask)
        self.gather_kv = None

    def load_model(self, checkpoints: str, use_tp: bool, rank_group=None, group = None):
        self.model: Transformer = load_model(checkpoint_path=checkpoints, device=self.device, precision=self.dtype, use_tp= use_tp, rank_group=rank_group, group = group)

    @torch.inference_mode()
    def setup_caches(self, max_batch_size: int = 1, max_seq_length: int = 2048, max_depth=1):
        self.max_length = max_seq_length
        self.batch_size = max_batch_size
        with torch.device(self.device):
                self.model.setup_caches(max_batch_size=max_batch_size, max_seq_length=max_seq_length)
        self.gather_kv = cuda_graph_for_gather_kv(self.device, max_batch_size, max_depth, self.model)   

    def compile(self, encode=False):
        import torch._dynamo.config
        import torch._inductor.config
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future
        for key in self.model_forward.keys():
            self.model_forward[key] = torch.compile(self.model_forward[key], mode="reduce-overhead", fullgraph=True)
        if encode:
             self.prefill = torch.compile(self.prefill, mode="reduce-overhead", fullgraph=True)      
             
    @torch.inference_mode()
    # @sdpa_kernel([SDPBackend.MATH])
    # @sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION])
    @sdpa_kernel([SDPBackend.FLASH_ATTENTION])
    def inference(self, input_ids: torch.LongTensor, position_ids: torch.LongTensor, storage_ids: torch.LongTensor, attention_mask: torch.Tensor):
            dec_len = input_ids.shape[1]
            return self.model_forward[dec_len](
                model=self.model, 
                x=input_ids.clone(),
                input_pos=position_ids.clone(),
                cache_pos=storage_ids.clone(),
                attention_mask=attention_mask.clone()).clone() if dec_len in self.model_forward.keys() else self.model.forward(input_ids, position_ids, storage_ids, attention_mask).clone()
    
    @torch.inference_mode()
    def encode(self, input_ids: torch.LongTensor, position_ids: torch.LongTensor, storage_ids: torch.LongTensor, attention_mask: torch.Tensor):
            return self.prefill(
                 model=self.model, 
                 x=input_ids.clone(),
                 input_pos=position_ids.clone(),
                 cache_pos=storage_ids.clone(),
                 attention_mask=attention_mask.clone()).clone()            
    
    @torch.inference_mode()
    def gather_kv_incremental(self, indices: Tensor, offset:Tensor):
        self.gather_kv(indices, offset)
    
    @torch.inference_mode()
    def clear_kv(self):
        for b in self.model.layers:
            b.attention.kv_cache.k_cache.zero_()
            b.attention.kv_cache.v_cache.zero_()

    

