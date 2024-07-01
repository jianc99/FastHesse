import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from FastHesse.Engine.model_pipe import Transformer_pipe
from torch.nn.attention import SDPBackend, sdpa_kernel
from FastHesse.Engine.utils import load_model_pipe, cuda_graph_for_gather_kv_1, cuda_graph_for_gather_kv_2

class LMBackend:
    def __init__(self, dtype = torch.bfloat16, device: str = "cuda:0", dec_list: list = [1]) -> None:
        self.dtype = dtype
        self.device = device
        self.model_forward1 = {}
        self.model_forward2 = {}
        for dec_len in dec_list:
            if dec_len == 0: continue
            self.model_forward1[dec_len] = lambda model, x, input_pos, cache_pos, attention_mask: model.forward_1(x, input_pos, cache_pos, attention_mask)
        for dec_len in dec_list:
            if dec_len == 0: continue
            self.model_forward2[dec_len] = lambda model, x, input_pos, cache_pos, attention_mask: model.forward_2(x, input_pos, cache_pos, attention_mask)
        self.prefill_1 = lambda model, x, input_pos, cache_pos, attention_mask: model.forward_1(x, input_pos, cache_pos, attention_mask)
        self.prefill_2 = lambda model, x, input_pos, cache_pos, attention_mask: model.forward_2(x, input_pos, cache_pos, attention_mask)
        self.gather_kv_1 = None
        self.gather_kv_2 = None

    def load_model(self, checkpoints: str, use_tp: bool, rank_group=None, process_group = None):
        self.model: Transformer_pipe = load_model_pipe(checkpoint_path=checkpoints, device=self.device, precision=self.dtype, use_tp= use_tp, rank_group=rank_group, process_group = process_group)

    @torch.inference_mode()
    def setup_caches(self, max_batch_size: int = 1, max_seq_length: int = 2048, max_depth=1):
        self.max_length = max_seq_length
        self.batch_size = max_batch_size
        with torch.device(self.device):
            self.model.setup_caches(max_batch_size=max_batch_size, max_seq_length=max_seq_length)
        self.gather_kv_1 = cuda_graph_for_gather_kv_1(self.device, max_batch_size, max_depth, self.model)
        self.gather_kv_2 = cuda_graph_for_gather_kv_2(self.device, max_batch_size, max_depth, self.model)   

    def compile(self, encode=False):
        import torch._dynamo.config
        import torch._inductor.config
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future
        # torch._inductor.config.assert_indirect_indexing = False
        for key in self.model_forward1.keys():
            self.model_forward1[key] = torch.compile(self.model_forward1[key], mode="reduce-overhead", fullgraph=True)
        for key in self.model_forward2.keys():
            self.model_forward2[key] = torch.compile(self.model_forward2[key], mode="reduce-overhead", fullgraph=True)
        if encode:
             self.prefill_1 = torch.compile(self.prefill_1, mode="reduce-overhead", fullgraph=True)
             self.prefill_2 = torch.compile(self.prefill_2, mode="reduce-overhead", fullgraph=True)      
             
    @torch.inference_mode()
    @sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION])
    def inference(self, input_ids: torch.LongTensor, position_ids: torch.LongTensor, storage_ids: torch.LongTensor, attention_mask: torch.Tensor, idx):
            if idx == 0:
                dec_len = input_ids.shape[1]
                return self.model_forward1[dec_len](
                    model=self.model, 
                    x=input_ids.clone(),
                    input_pos=position_ids.clone(),
                    cache_pos=storage_ids.clone(),
                    attention_mask=attention_mask.clone()).clone() if dec_len in self.model_forward1.keys() else self.model.forward_1(input_ids, position_ids, storage_ids, attention_mask).clone()
            elif idx == 1:
                dec_len = input_ids.shape[1]
                return self.model_forward2[dec_len](
                    model=self.model, 
                    x=input_ids.clone(),
                    input_pos=position_ids.clone(),
                    cache_pos=storage_ids.clone(),
                    attention_mask=attention_mask.clone()).clone() if dec_len in self.model_forward2.keys() else self.model.forward_1(input_ids, position_ids, storage_ids, attention_mask).clone()
    
    @torch.inference_mode()
    def encode(self, input_ids: torch.LongTensor, position_ids: torch.LongTensor, storage_ids: torch.LongTensor, attention_mask: torch.Tensor, idx):
            if idx == 0:
                return self.prefill_1(
                 model=self.model, 
                 x=input_ids.clone(),
                 input_pos=position_ids.clone(),
                 cache_pos=storage_ids.clone(),
                 attention_mask=attention_mask.clone()).clone()
            elif idx == 1:
                return self.prefill_2(
                 model=self.model, 
                 x=input_ids.clone(),
                 input_pos=position_ids.clone(),
                 cache_pos=storage_ids.clone(),
                 attention_mask=attention_mask.clone()).clone()      
    
    @torch.inference_mode()
    def gather_kv_incremental(self, indices: list[int], offset:int, idx):
        if idx == 0:
            self.gather_kv_1(indices, offset)
        elif idx == 1:
            self.gather_kv_2(indices, offset)
    
    @torch.inference_mode()
    def clear_kv(self):
        for b in self.model.layers:
            b.attention.kv_cache_1.k_cache.zero_()
            b.attention.kv_cache_1.v_cache.zero_()
            b.attention.kv_cache_2.k_cache.zero_()
            b.attention.kv_cache_2.v_cache.zero_()

    

