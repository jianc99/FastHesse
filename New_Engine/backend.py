import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from FastHesse.New_Engine.model import Transformer
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel
from FastHesse.New_Engine.utlis import load_model, model_forward, prefill, model_forward_branch

# class LMBackend:
#     def __init__(self, dtype = torch.bfloat16, device: str = "cuda:0") -> None:
#         self.dtype = dtype
#         self.device = device
#         self.model_forward = model_forward
#         self.prefill = prefill        

#     def load_model(self, checkpoints: str, use_tp: bool, rank_group=None):
#         self.model: Transformer = load_model(checkpoint_path=checkpoints, device=self.device, precision=self.dtype, use_tp= use_tp, rank_group=rank_group)   

#     def compile(self, encode=False):
#         import torch._dynamo.config
#         import torch._inductor.config
#         torch._inductor.config.coordinate_descent_tuning = True
#         torch._inductor.config.triton.unique_kernel_names = True
#         torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future
#         self.model_forward = torch.compile(model_forward, mode="reduce-overhead", fullgraph=True)
#         if encode:
#              self.prefill = torch.compile(prefill, mode="reduce-overhead", fullgraph=True)      
             
#     @torch.inference_mode()
#     @sdpa_kernel([SDPBackend.MATH])
#     def inference(self, input_ids: torch.LongTensor, position_ids: torch.LongTensor, storage_ids: torch.LongTensor, attention_mask: torch.Tensor):
#             return self.model_forward(
#                  model=self.model, 
#                  x=input_ids,
#                  input_pos=position_ids,
#                  cache_pos=storage_ids,
#                  attention_mask=attention_mask)
    
#     @torch.inference_mode()
#     def encode(self, input_ids: torch.LongTensor, position_ids: torch.LongTensor, storage_ids: torch.LongTensor, attention_mask: torch.Tensor):
#             return self.prefill(
#                  model=self.model, 
#                  x=input_ids,
#                  input_pos=position_ids,
#                  cache_pos=storage_ids,
#                  attention_mask=attention_mask)            
    
#     @torch.inference_mode()
#     def setup_caches(self, max_batch_size: int = 1, max_seq_length: int = 2048):
#             with torch.device(self.device):
#                  self.model.setup_caches(max_batch_size=max_batch_size, max_seq_length=max_seq_length)

#     @torch.inference_mode()
#     def gather_kv_incremental(self, indices: list[int], offset:int, batch_idx=None):
#         if batch_idx == None:
#             for b in self.model.layers:
#                 b.attention.kv_cache.k_cache[..., offset:offset + len(indices), :] = b.attention.kv_cache.k_cache[..., indices, :]
#                 b.attention.kv_cache.v_cache[..., offset:offset + len(indices), :] = b.attention.kv_cache.v_cache[..., indices, :]
#                 b.attention.kv_cache.k_cache[..., offset + len(indices):, :] = 0.0
#                 b.attention.kv_cache.v_cache[..., offset + len(indices):, :] = 0.0
#         else:
#              for b in self.model.layers:
#                 b.attention.kv_cache.k_cache[batch_idx, :, offset:offset + len(indices), :] = b.attention.kv_cache.k_cache[batch_idx, :, indices, :]
#                 b.attention.kv_cache.v_cache[batch_idx, :, offset:offset + len(indices), :] = b.attention.kv_cache.v_cache[batch_idx, :, indices, :]

#                 b.attention.kv_cache.k_cache[batch_idx, :, offset + len(indices):, :] = 0.0
#                 b.attention.kv_cache.v_cache[batch_idx, :, offset + len(indices):, :] = 0.0
    
#     @torch.inference_mode()
#     def clear_kv(self):
#          for b in self.model.layers:
#             b.attention.kv_cache.k_cache.zero_()
#             b.attention.kv_cache.v_cache.zero_()

# class LMBackendDraft:
#     def __init__(self, dtype = torch.bfloat16, device: str = "cuda:0") -> None:
#         self.dtype = dtype
#         self.device = device
#         self.model_forward = model_forward
#         self.model_forward_branch = model_forward_branch
#         self.prefill = prefill        

#     def load_model(self, checkpoints: str, use_tp: bool, rank_group=None):
#         self.model: Transformer = load_model(checkpoint_path=checkpoints, device=self.device, precision=self.dtype, use_tp= use_tp, rank_group=rank_group)   

#     def compile(self, encode=False):
#         import torch._dynamo.config
#         import torch._inductor.config
#         torch._inductor.config.coordinate_descent_tuning = True
#         torch._inductor.config.triton.unique_kernel_names = True
#         torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future
#         self.model_forward = torch.compile(model_forward, mode="reduce-overhead", fullgraph=True)
#         self.model_forward_branch = torch.compile(model_forward_branch, mode="reduce-overhead", fullgraph=True)
#         if encode:
#              self.prefill = torch.compile(prefill, mode="reduce-overhead", fullgraph=True)      
             
#     @torch.inference_mode()
#     @sdpa_kernel([SDPBackend.MATH])
#     def inference(self, input_ids: torch.LongTensor, position_ids: torch.LongTensor, storage_ids: torch.LongTensor, attention_mask: torch.Tensor):
#             return self.model_forward(
#                  model=self.model, 
#                  x=input_ids,
#                  input_pos=position_ids,
#                  cache_pos=storage_ids,
#                  attention_mask=attention_mask)
    
#     @torch.inference_mode()
#     @sdpa_kernel([SDPBackend.MATH])
#     def inference_branch(self, input_ids: torch.LongTensor, position_ids: torch.LongTensor, storage_ids: torch.LongTensor, attention_mask: torch.Tensor):
#             return self.model_forward_branch(
#                  model=self.model, 
#                  x=input_ids,
#                  input_pos=position_ids,
#                  cache_pos=storage_ids,
#                  attention_mask=attention_mask)
    
#     @torch.inference_mode()
#     def encode(self, input_ids: torch.LongTensor, position_ids: torch.LongTensor, storage_ids: torch.LongTensor, attention_mask: torch.Tensor):
#             return self.prefill(
#                  model=self.model, 
#                  x=input_ids,
#                  input_pos=position_ids,
#                  cache_pos=storage_ids,
#                  attention_mask=attention_mask)            
    
#     @torch.inference_mode()
#     def setup_caches(self, max_batch_size: int = 1, max_seq_length: int = 2048):
#             with torch.device(self.device):
#                  self.model.setup_caches(max_batch_size=max_batch_size, max_seq_length=max_seq_length)

#     @torch.inference_mode()
#     def gather_kv_incremental(self, indices: list[int], offset:int, batch_idx=None):
#         if batch_idx == None:
#             for b in self.model.layers:
#                 b.attention.kv_cache.k_cache[..., offset:offset + len(indices), :] = b.attention.kv_cache.k_cache[..., indices, :]
#                 b.attention.kv_cache.v_cache[..., offset:offset + len(indices), :] = b.attention.kv_cache.v_cache[..., indices, :]
#                 b.attention.kv_cache.k_cache[..., offset + len(indices):, :] = 0.0
#                 b.attention.kv_cache.v_cache[..., offset + len(indices):, :] = 0.0
#         else:
#              for b in self.model.layers:
#                 b.attention.kv_cache.k_cache[batch_idx, :, offset:offset + len(indices), :] = b.attention.kv_cache.k_cache[batch_idx, :, indices, :]
#                 b.attention.kv_cache.v_cache[batch_idx, :, offset:offset + len(indices), :] = b.attention.kv_cache.v_cache[batch_idx, :, indices, :]

#                 b.attention.kv_cache.k_cache[batch_idx, :, offset + len(indices):, :] = 0.0
#                 b.attention.kv_cache.v_cache[batch_idx, :, offset + len(indices):, :] = 0.0
    
#     @torch.inference_mode()
#     def clear_kv(self):
#          for b in self.model.layers:
#             b.attention.kv_cache.k_cache.zero_()
#             b.attention.kv_cache.v_cache.zero_()


class LMBackend:
    def __init__(self, dtype = torch.bfloat16, device: str = "cuda:0", dec_list: list = [1]) -> None:
        self.dtype = dtype
        self.device = device
        self.model_forward = {}
        for dec_len in dec_list:
            self.model_forward[dec_len] = lambda model, x, input_pos, cache_pos, attention_mask : model(x, input_pos, cache_pos, attention_mask)
        self.prefill = lambda model, x, input_pos, cache_pos, attention_mask : model(x, input_pos, cache_pos, attention_mask)      

    def load_model(self, checkpoints: str, use_tp: bool, rank_group=None):
        self.model: Transformer = load_model(checkpoint_path=checkpoints, device=self.device, precision=self.dtype, use_tp= use_tp, rank_group=rank_group)   

    def compile(self, encode=False):
        import torch._dynamo.config
        import torch._inductor.config
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future
        for key in self.model_forward.keys():
            self.model_forward[key] = torch.compile(self.model_forward[key], mode="reduce-overhead", fullgraph=True)
        if encode:
             self.prefill = torch.compile(self.prefill, mode="reduce-overhead", fullgraph=True, dynamic=True)      
             
    @torch.inference_mode()
    @sdpa_kernel([SDPBackend.MATH])
    def inference(self, input_ids: torch.LongTensor, position_ids: torch.LongTensor, storage_ids: torch.LongTensor, attention_mask: torch.Tensor):
            dec_len = input_ids.shape[1]
            if dec_len in self.model_forward.keys():
                return self.model_forward[dec_len](
                    model=self.model, 
                    x=input_ids,
                    input_pos=position_ids,
                    cache_pos=storage_ids,
                    attention_mask=attention_mask)
            else:
                 return model_forward(
                    model=self.model, 
                    x=input_ids,
                    input_pos=position_ids,
                    cache_pos=storage_ids,
                    attention_mask=attention_mask)
    
    @torch.inference_mode()
    def encode(self, input_ids: torch.LongTensor, position_ids: torch.LongTensor, storage_ids: torch.LongTensor, attention_mask: torch.Tensor):
            return self.prefill(
                 model=self.model, 
                 x=input_ids,
                 input_pos=position_ids,
                 cache_pos=storage_ids,
                 attention_mask=attention_mask)            
    
    @torch.inference_mode()
    def setup_caches(self, max_batch_size: int = 1, max_seq_length: int = 2048):
            with torch.device(self.device):
                 self.model.setup_caches(max_batch_size=max_batch_size, max_seq_length=max_seq_length)

    @torch.inference_mode()
    def gather_kv_incremental(self, indices: list[int], offset:int, batch_idx=None):
        if batch_idx == None:
            self.model.kv_cache.k_cache[..., offset:offset + len(indices), :] = self.model.kv_cache.k_cache[..., indices, :]
            self.model.kv_cache.v_cache[..., offset:offset + len(indices), :] = self.model.kv_cache.v_cache[..., indices, :]
            self.model.kv_cache.k_cache[..., offset + len(indices):, :] = 0.0
            self.model.kv_cache.v_cache[..., offset + len(indices):, :] = 0.0
        else:
            self.model.kv_cache.k_cache[:, batch_idx, :, offset:offset + len(indices), :] = self.model.kv_cache.k_cache[:, batch_idx, :, indices, :]
            self.model.kv_cache.v_cache[:, batch_idx, :, offset:offset + len(indices), :] = self.model.kv_cache.v_cache[:, batch_idx, :, indices, :]
            self.model.kv_cache.k_cache[:, batch_idx, :, offset + len(indices):, :] = 0.0
            self.model.kv_cache.v_cache[:, batch_idx, :, offset + len(indices):, :] = 0.0
    
    @torch.inference_mode()
    def clear_kv(self):
         self.model.kv_cache.k_cache.zero_()
         self.model.kv_cache.v_cache.zero_()
        #  for b in self.model.layers:
        #     b.attention.kv_cache.k_cache.zero_()
        #     b.attention.kv_cache.v_cache.zero_()

    

