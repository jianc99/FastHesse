import torch
from torch.nn.functional import softmax
from .Tree import BatchTree
import time
from FastHesse.Engine.backend_pipe import LMBackend
from .utils import get_sampling_logits, ChildrenAccept
import torch.distributed as dist

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

        self.kv_offsets = None
        self.kv_indices = None

        tree_mask :torch.Tensor = self.grow_map["mask"].to(self.device)
        tree_mask = (tree_mask == 1).type(torch.bool)
        self.set_prefix(prefix=prefix)

        self.tree_size = self.grow_map["size"]
        self.tree_mask = tree_mask

        self.attn_mask = torch.full((self.batch_size, 1, self.tree_size ,self.max_length), 0, dtype=torch.bool, device=device)

        self.depth = self.grow_map["depth"].repeat(self.batch_size,1).to(self.device)

        self.tree_buffer = torch.zeros((self.batch_size, self.tree_size),device=self.device).long()
        self.draft_logits = torch.zeros((self.batch_size, self.tree_size, vocab_size), dtype=torch.bfloat16).to(self.device)

        self.target_token = torch.zeros((self.batch_size, self.tree_size), device=self.device).long()

        self.make_inference_para_for_first_itr(prefix.size(1))

        self.draft_model_engine.encode(input_ids=self.tokens[:, :prefix.size(1)], 
                            storage_ids=self.prefill_storage_ids, 
                            position_ids=self.prefill_position_ids,
                            attention_mask=self.prefill_mask, idx=self.idx)

        control_tensor = torch.tensor(idx,device=self.device)
        dist.broadcast(control_tensor, draft_rank0)
        dist.broadcast(prefix.to(self.device),draft_rank0)
        root_tokens = torch.zeros((self.batch_size,1),device=self.device).long()
        dist.broadcast(root_tokens, target_rank0)
        self.tree_buffer[:, 0] = root_tokens.squeeze(1)
        
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
                
        new_tokens_set = self.sampling_callables[grow_step](self.draft_logits[:, idx_list])
        self.tree_buffer[:, indices] = new_tokens_set[:, self.sample_gather_indices[grow_step]]
            
        if benchmark:
            torch.cuda.synchronize()
            t2 = time.time()
            x1 += (t2 - t1)
        
        draft_model_outputs = self.draft_model_engine.inference(
            input_ids = self.tree_buffer[:, indices],
            position_ids = self.position_ids[:, indices],
            attention_mask = self.attn_mask[:,:,indices,:],
            storage_ids=self.storage_ids[indices], idx=self.idx
        )
        self.draft_logits[:, indices] = draft_model_outputs[:, -total_branch:]

        if benchmark:
                    torch.cuda.synchronize()
                    t3 = time.time()
                    x2 += (t3 - t2)
        if benchmark:
            return n_branch_list, x1, x2
        return n_branch_list

    def request_target(self, benchmark = False):
        if benchmark:
            torch.cuda.synchronize()
            t1 = time.time()

        if self.idx == 0:
            control_tensor = torch.tensor(2, device=self.device)
            dist.broadcast(control_tensor, self.draft_rank0)
        elif self.idx == 1:
            control_tensor = torch.tensor(3, device=self.device)
            dist.broadcast(control_tensor, self.draft_rank0)
        dist.broadcast(self.tree_buffer, self.draft_rank0)

        if self.kv_indices != None and self.kv_offsets != None:
            dist.broadcast(self.kv_indices, self.draft_rank0)
            dist.broadcast(self.kv_offsets, self.draft_rank0)
            dist.broadcast(self.num_nodes, self.draft_rank0)

        if benchmark:
            torch.cuda.synchronize()
            t2 = time.time()
            return t2-t1
    
    def receive_result(self, benchmark=False):
         # Get target tokens from target group
        if benchmark:
            torch.cuda.synchronize()
            t1 = time.time()
        dist.broadcast(self.target_token, self.target_rank0)
        if benchmark:
            torch.cuda.synchronize()
            t2 = time.time()
            return t2-t1
        
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
    def verify(self, benchmark = False, other_terminate = False):
        if benchmark:
            torch.cuda.synchronize()
            t0 = time.time()
        # Compare the target token with draft tokens
        terminal = False
        self.kv_indices = torch.full((self.batch_size, self.draft_step),self.max_target_seq-1,device=self.device).long()
        self.kv_indices[:, 0] = self.max_length-self.tree_size
        self.kv_offsets = self.num_nodes.clone()
        bonus_tokens = torch.full((self.batch_size, 1), 0, device=self.device).long()
        
        for batch_idx in range(self.batch_size):
            accept_list = [0]
            seq_idx = 0
            while True:
                parent_id = accept_list[-1]
                pos = self.accept_step(parent_id=parent_id, idx=batch_idx)
                if pos != -1:
                    seq_idx+=1
                    accept_list.append(pos)
                    self.kv_indices[batch_idx, seq_idx] = self.max_length-self.tree_size+pos
                    if self.tree_buffer[batch_idx, pos] == 0 or self.tree_buffer[batch_idx, pos] == 2:
                        terminal = True
                        break
                else:
                    break
            accept_length = len(accept_list)
            self.tokens[batch_idx, self.num_nodes[batch_idx]:self.num_nodes[batch_idx]+accept_length] = self.tree_buffer[batch_idx, accept_list]
            self.num_nodes[batch_idx]+= accept_length
            bonus_token = self.target_token[batch_idx, accept_list[-1]].reshape(1)
            if (bonus_token == 2) or (bonus_token == 0): 
                terminal = True
            bonus_tokens[batch_idx] = bonus_token
        if benchmark:
            torch.cuda.synchronize()
            t1 = time.time()

# Check Number of Nodes + Bonus Token <= max_target_token
        if self.num_nodes.max()+ 1 >= self.max_target_seq:
            terminal = True

# Put Bonus tokens to the root of tree
        if not terminal and not other_terminate:
            self.tree_buffer[:, :1]= bonus_tokens

# Gather KV Cache
        if not terminal and not other_terminate:
            self.draft_model_engine.gather_kv_incremental(self.kv_indices, self.kv_offsets.view(-1,1), idx = self.idx)

        if not terminal and not other_terminate:
            if benchmark:
                torch.cuda.synchronize()
                t2 = time.time()
                return self.num_nodes, t1 - t0, t2-t1, terminal
            return self.num_nodes, terminal
# If terminal, put the bonus token into the overall token set        
        else:
             for i in range(self.batch_size):
                self.tokens[i, self.num_nodes[i]] = bonus_tokens[i]
             self.num_nodes += 1
             if benchmark:
                torch.cuda.synchronize()
                t2 = time.time()
                return self.num_nodes, t1 - t0, t2-t1, terminal
             return self.num_nodes, terminal
    
    def verbose(self):
        super().verbose()
    
    
    def construct_grow_map(self, benchmark = False):

        if benchmark:
            sample_time = 0
            compute_time = 0
            torch.cuda.synchronize()
            pre_start = time.time()
        self.prepare_for_next_iter()
        if benchmark:
            torch.cuda.synchronize()
            pre_end = time.time()
            compute_time += pre_end - pre_start
            
        for i in range(self.draft_step - 1):
                if benchmark:
                        _, t1, t2 = self.collective_grow_static(self.grow_map_roots_gpu[i], self.grow_map['branches'][i], benchmark=benchmark, grow_step=i)
                        sample_time += t1
                        compute_time += t2   
                else:
                        self.collective_grow_static(self.grow_map_roots_gpu[i], self.grow_map['branches'][i], grow_step=i)
                
        if benchmark:
            return sample_time, compute_time
        else:
            return None
    
    def prepare_for_next_iter(self):
        self.make_inference_para_for_next_itr()
        draft_model_outputs = self.draft_model_engine.inference(input_ids = self.tree_buffer[:,:1], 
                                                    storage_ids=self.storage_ids[:1],
                                                    position_ids=self.position_ids[:, :1],
                                                    attention_mask=self.attn_mask[:,:,:1,:], idx=self.idx
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
                 idx = 0,
                 ) -> None:
        super().__init__(device=device, max_length=max_length, batch_size=batch_size)
        self.max_target_seq = max_target_seq
        self.target_model_engine = target_model_engine
        self.temperature = temperature
        self.top_p = top_p
        self.grow_map = grow_map
        self.draft_step = len(self.grow_map["roots"])
        self.vocab_size = vocab_size

        self.draft_rank0=draft_rank0
        self.target_rank0=target_rank0
        self.idx = idx

        tree_mask :torch.Tensor = self.grow_map["mask"].to(self.device)
        tree_mask = (tree_mask == 1).type(torch.bool)
        self.set_prefix(prefix=prefix)

        self.tree_size = self.grow_map["size"]
        self.tree_mask = tree_mask

        self.kv_offsets = None
        self.kv_indices = None

        self.attn_mask = torch.full((self.batch_size, 1, self.tree_size ,self.max_length), 0, dtype=torch.bool, device=device)

        self.depth = self.grow_map["depth"].repeat(self.batch_size,1).to(self.device)

        self.tree_buffer = torch.zeros((self.batch_size, self.tree_size),device=self.device).long()

        self.make_inference_para_for_first_itr(prefix.size(1))

        output = self.target_model_engine.encode(input_ids=self.tokens[:, :prefix.size(1)], 
                            storage_ids=self.prefill_storage_ids, 
                            position_ids=self.prefill_position_ids,
                            attention_mask=self.prefill_mask, idx = self.idx)
        
        root_logits = output[:, -1:].clone()
        root_logits = get_sampling_logits(logits=root_logits, top_p=self.top_p, T=self.temperature, replicate=False)
        root_logits = softmax(root_logits / self.temperature, dim=-1)
        root_tokens = root_logits.view(-1, self.vocab_size).multinomial(num_samples=1).view(self.batch_size, 1).long()
        dist.broadcast(root_tokens, target_rank0)

        
    @torch.inference_mode()
    def verify(self, benchmark = False):
        # Receive the token tree and verified token information from draft group
        if benchmark:
            torch.cuda.synchronize()
            t0 = time.time()
        dist.broadcast(self.tree_buffer, self.draft_rank0)
        if self.kv_indices == None and self.kv_offsets == None:
            self.kv_indices = torch.full((self.batch_size, self.draft_step),self.max_target_seq-1, device=self.device).long()
            self.kv_offsets = self.num_nodes.clone()
        else:
            dist.broadcast(self.kv_indices, self.draft_rank0)
            dist.broadcast(self.kv_offsets, self.draft_rank0)
            dist.broadcast(self.num_nodes, self.draft_rank0)
            self.target_model_engine.gather_kv_incremental(self.kv_indices, self.kv_offsets.view(-1,1), self.idx)
            self.make_inference_para_for_next_itr()
        if benchmark:
            torch.cuda.synchronize()
            t1 = time.time()
        # Inference to get the target logits
        target_model_outputs = self.target_model_engine.inference(input_ids = self.tree_buffer, 
                                    position_ids =self.position_ids, attention_mask = self.attn_mask,
                                    storage_ids=self.storage_ids, idx = self.idx)
        if benchmark:
            torch.cuda.synchronize()
            t2 = time.time()
        # Sample to get the target tokens and send back to draft group
        self.target_logits :torch.FloatTensor = target_model_outputs[:, -self.tree_size:]
        self.target_logits = get_sampling_logits(logits=self.target_logits, top_p=self.top_p, T=self.temperature, replicate=False)
        self.target_logits = softmax(self.target_logits / self.temperature, dim=-1)
        self.target_token = self.target_logits.view(-1, self.vocab_size).multinomial(num_samples=1).view(self.batch_size, self.tree_size).long()
        dist.broadcast(self.target_token, self.target_rank0)
        if benchmark:
            torch.cuda.synchronize()
            t3 = time.time()
            return t1-t0, t2-t1, t3-t2
    
    def verbose(self):
        super().verbose()

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