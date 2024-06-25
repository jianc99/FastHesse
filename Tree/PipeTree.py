import torch
from torch.nn.functional import softmax
from .Tree import BatchTree
import time
from FastHesse.Engine.backend import LMBackend
from .utils import get_sampling_logits, ChildrenAccept

import torch.distributed as dist
def init_cache(model:LMBackend):
    num_layers = len(model.model.layers)
    dtype = model.model.layers[0].attention.kv_cache.k_cache.dtype
    device = model.model.layers[0].attention.kv_cache.k_cache.device
    cache_shape = model.model.layers[0].attention.kv_cache.k_cache.shape
    k_cache = torch.zeros((num_layers, cache_shape[0], cache_shape[1], cache_shape[2], cache_shape[3]), dtype=dtype, device=device)
    v_cache = torch.zeros((num_layers, cache_shape[0], cache_shape[1], cache_shape[2], cache_shape[3]), dtype=dtype, device=device)
    return k_cache, v_cache

@torch.inference_mode()
def copy_to_model(model:LMBackend, k_cache, v_cache):
     for index, b in enumerate(model.model.layers):
            b.attention.kv_cache.k_cache.copy_(k_cache[index])
            b.attention.kv_cache.v_cache.copy_(v_cache[index])

@torch.inference_mode()
def cuda_graph_for_copy_to_model(model, k_cache, v_cache, n_warmups=3, mempool=None):
    
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmups):
            copy_to_model(model, k_cache, v_cache)
        s.synchronize()
    torch.cuda.current_stream().wait_stream(s)
    time.sleep(1)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        copy_to_model(model, k_cache, v_cache)
    def run():
        graph.replay()
    return run


@torch.inference_mode()
def copy_from_model(model:LMBackend, k_cache, v_cache):
     for index, b in enumerate(model.model.layers):
            k_cache[index].copy_(b.attention.kv_cache.k_cache)
            v_cache[index].copy_(b.attention.kv_cache.v_cache)

@torch.inference_mode()
def gather_kv(k_cache, v_cache, indices, offset):
    indices_len = indices.shape[1]
    storage_ids = torch.arange(0, indices.shape[1], dtype=torch.long, device=k_cache.device)
    num_layers, batch_size, num_heads, seq_length, head_dim = k_cache.shape
    indices = indices.unsqueeze(0).unsqueeze(2).unsqueeze(4)
    indices = indices.expand(num_layers, -1, num_heads, -1, head_dim)
    new_k_cache = torch.gather(k_cache, 3, indices)
    new_v_cache = torch.gather(v_cache, 3, indices)
    # k_cache.index_copy_(
    #     dim=-2, index=storage_ids, source=new_k_cache
    # )
    # v_cache.index_copy_(
    #     dim=-2, index=storage_ids, source=new_v_cache
    # )
    k_cache[...,offset:offset+indices_len,:] = new_k_cache
    v_cache[...,offset:offset+indices_len,:] = new_v_cache

# @torch.inference_mode()
# def cuda_graph_for_gather_kv(
#                 device="cuda:0", 
#                 k_cache=None, v_cache=None, max_len=256,
#                 n_warmups=3, mempool=None):
    
#     static_indices = torch.full((k_cache.shape[1], max_len), 1, dtype=torch.long, device=device)
#     s = torch.cuda.Stream()
#     s.wait_stream(torch.cuda.current_stream())
#     with torch.cuda.stream(s):
#         for _ in range(n_warmups):
#             gather_kv(k_cache, v_cache, static_indices)
#         s.synchronize()
#     torch.cuda.current_stream().wait_stream(s)
#     time.sleep(1)
#     graph = torch.cuda.CUDAGraph()
#     with torch.cuda.graph(graph, pool=mempool):
#         gather_kv(k_cache, v_cache, static_indices)
#     def run(indices):
#         static_indices.copy_(indices)
#         graph.replay()
#     return run

class PipeTree_Draft(BatchTree):
    def __init__(self, 
                 draft_model_engine: LMBackend,
                 prefix :torch.LongTensor,
                 max_length = 256,
                 device :str = 'cpu',
                 max_target_seq = 256,
                 vocab_size = 32000,
                 batch_size = 1,
                 grow_map = None,
                 sampling_callables = None,
                 sample_gather_indices = None,
                 target_rank0=0,
                 draft_rank0=0,
                 idx=0,
                 ) -> None:
        super().__init__(device=device, max_length=max_length, batch_size=batch_size)
        self.max_target_seq = max_target_seq
        self.draft_model_engine = draft_model_engine
        self.grow_map = grow_map
        self.sampling_callables = sampling_callables
        self.sample_gather_indices = sample_gather_indices
        self.draft_step = len(self.grow_map["roots"])
        self.vocab_size = vocab_size
        self.grow_map_roots_gpu = []
        for x in self.grow_map["roots"]:
             self.grow_map_roots_gpu.append(torch.Tensor(x).to(self.device).long())
        self.Successors = self.grow_map["Successors"]

        self.target_rank0=target_rank0
        self.draft_rank0=draft_rank0
        self.idx = idx

        self.k_cache, self.v_cache = init_cache(draft_model_engine)
        # self.gather_kv = cuda_graph_for_gather_kv(self.device, self.k_cache, self.v_cache, self.max_target_seq)
        self.gather_kv=gather_kv

        tree_mask :torch.Tensor = self.grow_map["mask"].to(self.device)
        tree_mask = (tree_mask == 1).type(torch.bool)
        self.initialize(None)
        self.set_prefix(prefix=prefix)

        self.tree_size = self.grow_map["size"]
        self.tree_mask = tree_mask

        self.attn_mask = torch.full((self.batch_size, 1, self.tree_size ,self.max_length), 0, dtype=torch.bool, device=device)

        self.depth = self.grow_map["depth"].repeat(self.batch_size,1).to(self.device)

        self.tree_buffer = torch.zeros((self.batch_size, self.tree_size),device=self.device).long()
        self.draft_logits = torch.zeros((self.batch_size, self.tree_size, vocab_size), dtype=torch.bfloat16).to(self.device)

        self.make_inference_para_for_first_itr(prefix.size(1))

        # print(self.k_cache[0,0], self.v_cache[0,0])
        torch.cuda.synchronize()
        t1 = time.time()
        copy_to_model(model=draft_model_engine, k_cache=self.k_cache, v_cache=self.v_cache)
        torch.cuda.synchronize()
        t2 = time.time()

        self.draft_model_engine.encode(input_ids=self.tokens[:, :prefix.size(1)], 
                            storage_ids=self.prefill_storage_ids, 
                            position_ids=self.prefill_position_ids,
                            attention_mask=self.prefill_mask)
        torch.cuda.synchronize()
        t3 = time.time()
        copy_from_model(model=draft_model_engine, k_cache=self.k_cache, v_cache=self.v_cache)
        torch.cuda.synchronize()
        t4 = time.time()
        print(f"Draft copy:{t2-t1}{t4-t3}")

        # print(self.k_cache[0,0], self.v_cache[0,0])
        # time.sleep(1000)

        control_tensor = torch.tensor([idx],device=self.device)
        dist.broadcast(control_tensor, draft_rank0)
        dist.broadcast(prefix.to(self.device),draft_rank0)
        root_tokens = torch.zeros((self.batch_size,1),device=self.device).long()
        dist.broadcast(root_tokens, target_rank0)
        self.tree_buffer[:, 0] = root_tokens.squeeze(1)
        
        # self.prepare_for_next_iter()

    @torch.inference_mode()
    def collective_grow_static(self, idx_list, n_branch_list :list[int], benchmark=False, grow_step = None):
        
        if benchmark:
            x1 = 0.0
            x2 = 0.0
        
        total_branch = sum(n_branch_list)
        indices = self.grow_map_roots_gpu[grow_step+1]
        

        # Sample from stored logits
        if benchmark:
                torch.cuda.synchronize()
                t1 = time.time()
        for i in range(self.batch_size):
            new_tokens_set = self.sampling_callables[grow_step](self.draft_logits[i, idx_list])
            self.tree_buffer[i, indices] = new_tokens_set[self.sample_gather_indices[grow_step]]
            
        if benchmark:
                    torch.cuda.synchronize()
                    t2 = time.time()
                    x1 += (t2 - t1)
        
        draft_model_outputs = self.draft_model_engine.inference(
            input_ids = self.tree_buffer[:, indices],
            position_ids = self.position_ids[:, indices],
            attention_mask = self.attn_mask[:,:,indices,:],
            storage_ids=self.storage_ids[indices],
        )
        self.draft_logits[:, indices] = draft_model_outputs[:, -total_branch:]

        if benchmark:
                    torch.cuda.synchronize()
                    t3 = time.time()
                    x2 += (t3 - t2)
        if benchmark:
            return n_branch_list, x1, x2
        return n_branch_list

    def request_target(self):
        if self.idx == 0:
            control_tensor = torch.tensor([2], device=self.device)
            dist.broadcast(control_tensor, self.draft_rank0)
        elif self.idx == 1:
            control_tensor = torch.tensor([3], device=self.device)
            dist.broadcast(control_tensor, self.draft_rank0)
        dist.broadcast(self.tree_buffer, self.draft_rank0)
    
    def receive_result(self, benchmark=False):
         # Get verify result from target group
        if benchmark:
            torch.cuda.synchronize()
            t1 = time.time()
        terminal = torch.tensor([0], device=self.device)
        target_accept_list= torch.full((self.batch_size, self.draft_step),-1,device=self.device)
        bonus_tokens= torch.zeros((self.batch_size, 1),device=self.device).long()
        dist.broadcast(terminal, self.target_rank0)
        dist.broadcast(target_accept_list,self.target_rank0)
        dist.broadcast(bonus_tokens,self.target_rank0)
        self.terminal = terminal
        self.target_accept_list = target_accept_list
        self.bonus_tokens = bonus_tokens
        if benchmark:
            torch.cuda.synchronize()
            t2 = time.time()
            return t2-t1

        
    @torch.inference_mode()
    def verify(self, benchmark = False):
        # Get verify result from target group
        # copy_to_model(self.draft_model_engine, self.k_cache, self.v_cache)
        if benchmark:
            torch.cuda.synchronize()
            t1 = time.time()
        terminal = self.terminal
        target_accept_list = self.target_accept_list
        bonus_tokens = self.bonus_tokens
        if benchmark:
            torch.cuda.synchronize()
            t2 = time.time()
# Gather accepted tokens
        terminal = terminal.item()==1
        batch_accept_list=[]
        batch_accept_list_kv=[]
        kv_start = self.num_nodes.min()
        for i in range(self.batch_size):
             list=[]
             for j in range(self.draft_step):
                  if target_accept_list[i,j]!=-1:
                       list.append(target_accept_list[i,j].item())
                  else: break
             batch_accept_list.append(list)
             batch_accept_list_kv.append([self.max_length-self.tree_size+pos for pos in list])
             accept_length = len(list)
             self.tokens[i, self.num_nodes[i]:self.num_nodes[i]+accept_length] = self.tree_buffer[i, list]
             self.num_nodes[i]+=accept_length
        if benchmark:
            torch.cuda.synchronize()
            t3 = time.time()
# Gather KV Cache
        torch.cuda.synchronize()
        start = time.time()
        if not terminal:
            full_indices = torch.arange(kv_start, self.num_nodes.max(), device=self.device).unsqueeze(0).repeat(self.batch_size, 1).long()
            for i in range(self.batch_size):
                accept_list_kv=batch_accept_list_kv[i]
                accept_length = len(accept_list_kv)
                for j in range(len(accept_list_kv)):
                    full_indices[i,self.num_nodes[i]-kv_start-accept_length+j] = accept_list_kv[j]
            # self.draft_model_engine.gather_kv_incremental(indices, offsets.view(-1,1))
            self.gather_kv(self.k_cache, self.v_cache, full_indices, kv_start)
            self.tree_buffer[:, 0]= bonus_tokens.squeeze(1)
        torch.cuda.synchronize()
        end = time.time()
        print(f"Draft: {end-start}")
        # copy_from_model(self.draft_model_engine, self.k_cache, self.v_cache)
        if benchmark:
            torch.cuda.synchronize()
            t4 = time.time()
            return self.num_nodes, t2 - t1, t3-t2, t4 - t3, terminal
        return self.num_nodes, terminal
    
    
    def verbose(self):
        super().verbose()
    
    
    def construct_grow_map(self, benchmark = False):

        copy_to_model(model=self.draft_model_engine, k_cache=self.k_cache, v_cache=self.v_cache)

        self.prepare_for_next_iter()
        if benchmark:
            sample_time = 0
            compute_time = 0
        for i in range(self.draft_step - 1):
                if benchmark:
                        _, t1, t2 = self.collective_grow_static(self.grow_map_roots_gpu[i], self.grow_map['branches'][i], benchmark=benchmark, grow_step=i)
                        sample_time += t1
                        compute_time += t2   
                else:
                        self.collective_grow_static(self.grow_map_roots_gpu[i], self.grow_map['branches'][i], grow_step=i)
        
        copy_from_model(model=self.draft_model_engine, k_cache=self.k_cache, v_cache=self.v_cache)
        
        if benchmark:
            return sample_time, compute_time
        else:
            return None
    
    def prepare_for_next_iter(self):
        self.make_inference_para_for_next_itr()
        draft_model_outputs = self.draft_model_engine.inference(input_ids = self.tree_buffer[:,:1], 
                                                    storage_ids=self.storage_ids[:1],
                                                    position_ids=self.position_ids[:, :1],
                                                    attention_mask=self.attn_mask[:,:,:1,:],
                                                    )
        
        self.draft_logits[:, 0] = draft_model_outputs[:, -1]

    def make_inference_para_for_first_itr(self, prefix_len):
        # Prefill
         self.prefill_mask=self.full_attn_mask[:prefix_len][None, None, :, :].repeat(self.batch_size,1,1,1)
         self.prefill_storage_ids = torch.arange(prefix_len).to(self.device)
         self.prefill_position_ids = self.prefill_storage_ids.clone().repeat(self.batch_size, 1)

        # Draft Construct Tree and Target Verify
         self.attn_mask[:, :, :, -self.tree_size:] = self.tree_mask
         for idx in range(self.batch_size):
            self.attn_mask[idx, :, :, :prefix_len] = True
         self.position_ids = (self.grow_map["depth"].to(self.device) + prefix_len).repeat(self.batch_size, 1)
         self.storage_ids = torch.arange(self.max_length-self.tree_size, self.max_length).to(self.device)
    
    def make_inference_para_for_next_itr(self):
         self.position_ids = self.depth + self.num_nodes.view(-1,1)
         for idx in range(self.batch_size):
            self.attn_mask[idx, :, :, :self.num_nodes[idx]] = True


class PipeTree_Target(BatchTree):
    def __init__(self, 
                 target_model_engine: LMBackend,
                 prefix :torch.LongTensor,
                 temperature :float = 0.6,
                 top_p: float = 0.9,
                 max_length = 256,
                 device :str = 'cpu',
                 max_target_seq = 256,
                 vocab_size = 32000,
                 batch_size = 1,
                 grow_map = None,
                 target_rank0=0,
                 draft_rank0=0,
                 ) -> None:
        super().__init__(device=device, max_length=max_length, batch_size=batch_size)
        self.max_target_seq = max_target_seq
        self.target_model_engine = target_model_engine
        self.temperature = temperature
        self.top_p = top_p
        self.grow_map = grow_map
        self.draft_step = len(self.grow_map["roots"])
        self.vocab_size = vocab_size
        self.grow_map_roots_gpu = []
        for x in self.grow_map["roots"]:
             self.grow_map_roots_gpu.append(torch.Tensor(x).to(self.device).long())
        self.Successors = self.grow_map["Successors"]

        self.draft_rank0=draft_rank0
        self.target_rank0=target_rank0

        self.k_cache, self.v_cache = init_cache(target_model_engine)
        # self.gather_kv = cuda_graph_for_gather_kv(self.device, self.k_cache, self.v_cache, self.max_target_seq)
        self.gather_kv = gather_kv

        tree_mask :torch.Tensor = self.grow_map["mask"].to(self.device)
        tree_mask = (tree_mask == 1).type(torch.bool)
        self.initialize(None)
        self.set_prefix(prefix=prefix)

        self.tree_size = self.grow_map["size"]
        self.tree_mask = tree_mask

        self.attn_mask = torch.full((self.batch_size, 1, self.tree_size ,self.max_length), 0, dtype=torch.bool, device=device)

        self.depth = self.grow_map["depth"].repeat(self.batch_size,1).to(self.device)

        self.tree_buffer = torch.zeros((self.batch_size, self.tree_size),device=self.device).long()

        self.make_inference_para_for_first_itr(prefix.size(1))

        torch.cuda.synchronize()
        t1 = time.time()
        copy_to_model(model=target_model_engine, k_cache=self.k_cache, v_cache=self.v_cache)
        torch.cuda.synchronize()
        t2 = time.time()

        output = self.target_model_engine.encode(input_ids=self.tokens[:, :prefix.size(1)], 
                            storage_ids=self.prefill_storage_ids, 
                            position_ids=self.prefill_position_ids,
                            attention_mask=self.prefill_mask)
        torch.cuda.synchronize()
        t3 = time.time()
        copy_from_model(model=target_model_engine, k_cache=self.k_cache, v_cache=self.v_cache)
        torch.cuda.synchronize()
        t4 = time.time()
        print(f"Target copy:{t2-t1}{t4-t3}")
        
        root_logits = output[:, -1:].clone()
        root_logits = get_sampling_logits(logits=root_logits, top_p=self.top_p, T=self.temperature, replicate=False)
        root_logits = softmax(root_logits / self.temperature, dim=-1)
        root_tokens = root_logits.view(-1, self.vocab_size).multinomial(num_samples=1).view(self.batch_size, 1)
        dist.broadcast(root_tokens, target_rank0)
        
        self.prepare_for_next_iter()
    
    @torch.inference_mode()
    def accept_step(self, parent_id :int, idx) ->ChildrenAccept:
        logits_id = parent_id
        
        target_token = self.target_token[idx, logits_id]
        children = self.Successors[logits_id]
        if len(children) == 0:
            return -1
        
        for pos in children:
            token = self.tree_buffer[idx, pos]
            if token == target_token:
                return pos
        return -1

        
    @torch.inference_mode()
    def verify(self, benchmark = False):
        # Inference to get the target tokens
        if benchmark:
            torch.cuda.synchronize()
            t0 = time.time()
        dist.broadcast(self.tree_buffer, self.draft_rank0)
        if benchmark:
            torch.cuda.synchronize()
            t1 = time.time()
        copy_to_model(model=self.target_model_engine, k_cache=self.k_cache, v_cache=self.v_cache)
        target_model_outputs = self.target_model_engine.inference(input_ids = self.tree_buffer, 
                                    position_ids =self.position_ids, attention_mask = self.attn_mask,
                                    storage_ids=self.storage_ids)
        copy_from_model(model=self.target_model_engine, k_cache=self.k_cache, v_cache=self.v_cache)
        if benchmark:
            torch.cuda.synchronize()
            t2 = time.time()

        self.target_logits :torch.FloatTensor = target_model_outputs[:, -self.tree_size:]
        self.target_logits = get_sampling_logits(logits=self.target_logits, top_p=self.top_p, T=self.temperature, replicate=False)
        self.target_logits = softmax(self.target_logits / self.temperature, dim=-1)
        self.target_token = self.target_logits.view(-1, self.vocab_size).multinomial(num_samples=1).view(self.batch_size, self.tree_size)
# Compare the target token with draft tokens
        terminal = False
        batch_accept_list=[]
        batch_accept_list_kv= []
        kv_start = self.num_nodes.min()
        for batch_idx in range(self.batch_size):
            accept_list = [0]
            while True:
                parent_id = accept_list[-1]
                pos = self.accept_step(parent_id=parent_id, idx=batch_idx)
                if pos != -1:
                    accept_list.append(pos)
                    if self.tree_buffer[batch_idx, pos] == 0 or self.tree_buffer[batch_idx, pos] == 2:
                        terminal = True
                        break
                else:
                    break
            batch_accept_list.append(accept_list)
            accept_length = len(accept_list)
            batch_accept_list_kv.append([self.max_length-self.tree_size+pos for pos in accept_list])
            self.tokens[batch_idx, self.num_nodes[batch_idx]:self.num_nodes[batch_idx]+accept_length] = self.tree_buffer[batch_idx, accept_list]
            self.num_nodes[batch_idx]+= accept_length
# Check Bonus token
        bonus_tokens = torch.zeros((self.batch_size,1), device=self.device).long()
        if not terminal:
            for batch_idx in range(self.batch_size):
                accept_list = batch_accept_list[batch_idx]
                bonus_token = self.target_token[batch_idx, accept_list[-1]].reshape(1)
                if (bonus_token == 2) or (bonus_token == 0): 
                    terminal = True
                    break
                bonus_tokens[batch_idx, 0]= bonus_token

        if benchmark:
            torch.cuda.synchronize()
            t3 = time.time()
        torch.cuda.synchronize()
        start = time.time()
# Gather KV Cache
        if self.num_nodes.max() + 1 >= self.max_target_seq:
             terminal=True

        if not terminal:
            full_indices = torch.arange(kv_start, self.num_nodes.max(), device=self.device).unsqueeze(0).repeat(self.batch_size, 1).long()
            for i in range(self.batch_size):
                accept_list_kv=batch_accept_list_kv[i]
                accept_length = len(accept_list_kv)
                for j in range(len(accept_list_kv)):
                    full_indices[i,self.num_nodes[i]-kv_start-accept_length+j] = accept_list_kv[j]
            self.gather_kv(self.k_cache, self.v_cache, full_indices, kv_start)
        # copy_from_model(model=self.target_model_engine, k_cache=self.k_cache, v_cache=self.v_cache)
        torch.cuda.synchronize()
        end = time.time()
        print(f"Target: {end-start}")
        target_accept_list = torch.full((self.batch_size, self.draft_step),-1,device=self.device)
        for i in range(self.batch_size):
             accept_list=batch_accept_list[i]
             for j in range(len(accept_list)):
                  target_accept_list[i,j] = accept_list[j]
        if terminal:
             terminate = torch.tensor([1], device=self.device)
             dist.broadcast(terminate, self.target_rank0)
        else:
             terminate = torch.tensor([0], device=self.device)
             dist.broadcast(terminate, self.target_rank0)
        dist.broadcast(target_accept_list, self.target_rank0)
        dist.broadcast(bonus_tokens, self.target_rank0)

        if not terminal:
            if benchmark:
                torch.cuda.synchronize()
                t4 = time.time()
                self.prepare_for_next_iter()
                return self.num_nodes, t1-t0, t2 - t1, t3-t2, t4 - t3, terminal
            self.prepare_for_next_iter()
            return self.num_nodes, terminal
            
        else:
             if benchmark:
                torch.cuda.synchronize()
                t4 = time.time()
                return self.num_nodes,t1-t0, t2 - t1, t3-t2, t4 - t3, terminal
             return self.num_nodes, terminal 
    
    def verbose(self):
        super().verbose()
    
    def prepare_for_next_iter(self):
        self.make_inference_para_for_next_itr()

    def make_inference_para_for_first_itr(self, prefix_len):
        # Prefill
         self.prefill_mask=self.full_attn_mask[:prefix_len][None, None, :, :].repeat(self.batch_size,1,1,1)
         self.prefill_storage_ids = torch.arange(prefix_len).to(self.device)
         self.prefill_position_ids = self.prefill_storage_ids.clone().repeat(self.batch_size, 1)

        # Draft Construct Tree and Target Verify
         self.attn_mask[:, :, :, -self.tree_size:] = self.tree_mask
         for idx in range(self.batch_size):
            self.attn_mask[idx, :, :, :prefix_len] = True
         self.position_ids = (self.grow_map["depth"].to(self.device) + prefix_len).repeat(self.batch_size, 1)
         self.storage_ids = torch.arange(self.max_length-self.tree_size, self.max_length).to(self.device)
    
    def make_inference_para_for_next_itr(self):
         self.position_ids = self.depth + self.num_nodes.view(-1,1)
         for idx in range(self.batch_size):
            self.attn_mask[idx, :, :, :self.num_nodes[idx]] = True