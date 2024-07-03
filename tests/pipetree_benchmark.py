import sys
sys.path.append("..")
import argparse
import time
import torch
import torch.distributed as dist
from pathlib import Path
from transformers import DataCollatorForLanguageModeling, AutoTokenizer
from FastHesse.Tree.PipeTree import PipeTree_Draft, PipeTree_Target
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from FastHesse.Engine.backend_pipe import LMBackend
from FastHesse.Data.data_converter import convert_wiki_dataset, convert_cnn_dataset, convert_c4_dataset
from FastHesse.Tree.utils import cuda_graph_for_sampling_argmax_batch
from FastHesse.Engine.utils import setup_seed
from FastHesse.Engine.tp_pipe import init_dist

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=Path, default=Path("checkpoints/meta-llama/Llama-2-7b-hf/model.pth"), help='model')
parser.add_argument('--target', type=Path, default=Path("checkpoints/meta-llama/Llama-2-70b-hf/model.pth"), help='target model')
parser.add_argument('--growmap', type=str, default="1.3b-70b_tree.pt", help='growmap path')
parser.add_argument('--dataset', type=str, default="cnn", help='dataset path')
parser.add_argument('--start', type=int, default=0, help='start')
parser.add_argument('--end', type=int, default=200, help='end')
parser.add_argument('--T', type=float, default=0.6, help='temperature')
parser.add_argument('--P', type=float, default=0.9, help='top_p')
parser.add_argument('--B', type=int, default=16, help='batch_size')
parser.add_argument('--M', type=int, default=256, help='max length')
parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument('--Mode', type=str, default="fast")
parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')

# Target model information
parser.add_argument('--target_group', nargs='+', type=int, help='Target group of ranks')
# Draft model information
parser.add_argument('--draft_group', nargs='+', type=int, help='Target group of ranks')
args = parser.parse_args()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
global print
global_rank, draft_group, target_group = init_dist(args.draft_group, args.target_group)
setup_seed(args.seed)
if not (global_rank == args.target_group[0] or global_rank == args.draft_group[0]):
    print = lambda *args, **kwargs: None

global_rank=dist.get_rank()
BATCH_SIZE = args.B

def simulation_fast(draft_model: LMBackend, dataloader: DataLoader, max_length=512, grow_map=None, sampling_callables = None, sample_gather_indices = None, target_rank0=0, draft_rank0=0):
    num_eval_steps = len(dataloader)
    num_decoding_steps = 0
    num_large_model_steps = 0
    total_time = 0.0
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=False)
    batch_tree_1 = None
    batch_tree_2 = None
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader), total=num_eval_steps):
            input_ids = batch['input_ids'][..., :128]
            labels = batch['labels'][..., :128]
            terminate = False
            if (labels[:, -1] == -100)._is_any_true(): terminate = True
            mini_batch_1 = input_ids[:BATCH_SIZE//2]
            mini_batch_2 = input_ids[BATCH_SIZE//2:]
            batch_tree_1 = PipeTree_Draft(draft_model_engine=draft_model, prefix=mini_batch_1, max_length=max_length, device=DEVICE, batch_size=BATCH_SIZE//2, grow_map=grow_map, sampling_callables=sampling_callables, sample_gather_indices= sample_gather_indices, target_rank0=target_rank0, draft_rank0=draft_rank0, idx=0, max_target_seq=args.M)
            batch_tree_2 = PipeTree_Draft(draft_model_engine=draft_model, prefix=mini_batch_2, max_length=max_length, device=DEVICE, batch_size=BATCH_SIZE//2, grow_map=grow_map, sampling_callables=sampling_callables, sample_gather_indices= sample_gather_indices, target_rank0=target_rank0, draft_rank0=draft_rank0, idx=1, max_target_seq=args.M)
            num_nodes = torch.zeros(BATCH_SIZE,device=DEVICE).long()
            
            batch_tree_1.construct_grow_map()
            batch_tree_1.request_target()
            batch_tree_2.construct_grow_map()
            torch.cuda.synchronize()
            t1 = time.time()
            while terminate == False:
                batch_tree_1.receive_result()
                num_large_model_steps+=1
                batch_tree_2.request_target()
                num_nodes[:BATCH_SIZE//2], terminate = batch_tree_1.verify()
                if terminate == True:
                    batch_tree_2.receive_result()
                    num_large_model_steps+=1
                    num_nodes[BATCH_SIZE//2:], terminate = batch_tree_2.verify(other_terminate=True)
                    break

                batch_tree_1.construct_grow_map()
                batch_tree_2.receive_result()
                num_large_model_steps+=1
                batch_tree_1.request_target()
                num_nodes[BATCH_SIZE//2:], terminate = batch_tree_2.verify()
                if terminate == True:
                    batch_tree_1.receive_result()
                    num_large_model_steps+=1
                    num_nodes[:BATCH_SIZE//2], terminate = batch_tree_1.verify(other_terminate=True)
                    break
                batch_tree_2.construct_grow_map()

            torch.cuda.synchronize()
            t2 = time.time()
            num_decoding_steps += (num_nodes.sum() - input_ids.shape[1]*BATCH_SIZE)
            total_time += (t2 - t1)
            for i in range(BATCH_SIZE//2):
                print(tokenizer.decode(batch_tree_1.tokens[i,:batch_tree_1.num_nodes[i]]))
            for i in range(BATCH_SIZE//2):
                print(tokenizer.decode(batch_tree_2.tokens[i,:batch_tree_2.num_nodes[i]]))
            print("total time :{:.5f}s, time per iter :{:.5f}s, decoding step: {}, large model step: {}".format(total_time, total_time / num_large_model_steps, num_decoding_steps, num_large_model_steps))
            if step == 0:
                num_decoding_steps = 0
                num_large_model_steps = 0
                total_time = 0.0
            dist.barrier(draft_group)
        control_tensor = torch.tensor(4,device=DEVICE)
        dist.broadcast(control_tensor,draft_rank0)
    return num_decoding_steps / num_large_model_steps

def simulation_benchmark(draft_model: LMBackend, dataloader: DataLoader, max_length=512, grow_map=None, sampling_callables = None, sample_gather_indices = None, target_rank0=0, draft_rank0=0):
    num_eval_steps = len(dataloader)
    num_decoding_steps = 0
    num_large_model_steps = 0
    speculate_time = 0.0
    verify_time = 0.0
    accept_loop = 0.0
    kv_select = 0.0
    sample_time = 0.0
    small_model_compute = 0.0
    batch_tree_1 = None
    batch_tree_2 = None
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader), total=num_eval_steps):
            input_ids = batch['input_ids'][..., :128]
            labels = batch['labels'][..., :128]
            terminate = False
            if (labels[:, -1] == -100)._is_any_true(): terminate = True
            mini_batch_1 = input_ids[:BATCH_SIZE//2]
            mini_batch_2 = input_ids[BATCH_SIZE//2:]
            batch_tree_1 = PipeTree_Draft(draft_model_engine=draft_model, prefix=mini_batch_1, max_length=max_length, device=DEVICE, batch_size=BATCH_SIZE//2, grow_map=grow_map, sampling_callables=sampling_callables, sample_gather_indices= sample_gather_indices, target_rank0=target_rank0, draft_rank0=draft_rank0, idx=0, max_target_seq=args.M)
            batch_tree_2 = PipeTree_Draft(draft_model_engine=draft_model, prefix=mini_batch_2, max_length=max_length, device=DEVICE, batch_size=BATCH_SIZE//2, grow_map=grow_map, sampling_callables=sampling_callables, sample_gather_indices= sample_gather_indices, target_rank0=target_rank0, draft_rank0=draft_rank0, idx=1, max_target_seq=args.M)
            num_nodes = torch.zeros(BATCH_SIZE,device=DEVICE).long()
            
            batch_tree_1.construct_grow_map()
            batch_tree_1.request_target()
            batch_tree_2.construct_grow_map()

            while terminate == False:
                batch_tree_1.receive_result()
                num_large_model_steps+=1
                batch_tree_2.request_target()

                torch.cuda.synchronize()
                t0 = time.time()
                num_nodes[:BATCH_SIZE//2], x, y, terminate = batch_tree_1.verify(benchmark=True)
                torch.cuda.synchronize()
                t1 = time.time()
                verify_time += t1-t0
                accept_loop += x
                kv_select +=y

                if terminate == True:
                    batch_tree_2.receive_result()
                    num_large_model_steps+=1
                    torch.cuda.synchronize()
                    t9 = time.time()
                    num_nodes[BATCH_SIZE//2:],e, f, terminate = batch_tree_2.verify(benchmark=True, other_terminate=True)
                    torch.cuda.synchronize()
                    t10 = time.time()
                    verify_time+=t10-t9
                    accept_loop += e
                    kv_select +=f
                    break

                torch.cuda.synchronize()
                t2 = time.time()
                a, b = batch_tree_1.construct_grow_map(benchmark=True)  
                torch.cuda.synchronize()
                t3 = time.time()    
                speculate_time += t3-t2
                sample_time+=a
                small_model_compute+=b

                batch_tree_2.receive_result()
                num_large_model_steps+=1
                batch_tree_1.request_target()

                torch.cuda.synchronize()
                t4 = time.time()
                num_nodes[BATCH_SIZE//2:],z, w, terminate = batch_tree_2.verify(benchmark=True)
                torch.cuda.synchronize()
                t5 = time.time()
                verify_time += t5-t4
                accept_loop += z
                kv_select += w

                if terminate == True:
                    batch_tree_1.receive_result()
                    num_large_model_steps+=1
                    torch.cuda.synchronize()
                    t11 = time.time()
                    num_nodes[:BATCH_SIZE//2], g, h, terminate = batch_tree_1.verify(benchmark=True, other_terminate=True)
                    torch.cuda.synchronize()
                    t12 = time.time()
                    verify_time += t12-t11
                    accept_loop += g
                    kv_select += h
                    break

                torch.cuda.synchronize()
                t6 = time.time()
                c, d = batch_tree_2.construct_grow_map(benchmark=True)
                torch.cuda.synchronize()
                t7 = time.time()
                speculate_time+=t7-t6
                sample_time+=c
                small_model_compute+=d
            num_decoding_steps += (num_nodes.sum() - input_ids.shape[1]*BATCH_SIZE)
            print("total generated tokens: {}".format(num_decoding_steps), "large model steps: {}".format(num_large_model_steps), "avg num of tokens generate per batch: {}".format(num_decoding_steps / num_large_model_steps / (BATCH_SIZE//2)))
            print("speculate time: {}".format(speculate_time / num_large_model_steps),  "verify time: {}".format(verify_time / num_large_model_steps))
            print("accept loop: {}".format(accept_loop / num_large_model_steps), "kv select: {}".format(kv_select / num_large_model_steps))
            print("small model run: {}".format(small_model_compute / num_large_model_steps) , "sample time: {}".format(sample_time / num_large_model_steps))
            if step == 0:
                num_decoding_steps = 0
                num_large_model_steps = 0
                speculate_time = 0.0
                verify_time = 0.0
                accept_loop = 0.0
                kv_select = 0.0
                sample_time = 0.0
                small_model_compute = 0.0
            dist.barrier(draft_group)
        control_tensor = torch.tensor(4,device=DEVICE)
        dist.broadcast(control_tensor,draft_rank0)

if global_rank in args.target_group:
    use_tp = len(args.target_group)>1
    dist.barrier()
    path = args.growmap
    grow_map = torch.load(path)
    tree_size = grow_map["size"]
    draft_step = len(grow_map["roots"])

    MAX_LEN = args.M + tree_size
    TARGET_MODEL_CHECKPOINT = args.target
    DTYPE = torch.bfloat16

    cg_list_target = [tree_size]

    target_model = LMBackend(dtype=DTYPE, device=DEVICE, dec_list=cg_list_target)
    target_model.load_model(TARGET_MODEL_CHECKPOINT, use_tp=use_tp, rank_group = args.target_group, process_group = target_group)
    if args.compile:
        target_model.compile()
    target_model.setup_caches(max_batch_size=BATCH_SIZE//2, max_seq_length=MAX_LEN, max_depth=draft_step)
    target_rank0 = args.target_group[0]
    draft_rank0 = args.draft_group[0]
    mini_batch_1_tree = None
    mini_batch_2_tree = None
    if args.Mode == "benchmark":
        receive_gather_time = 0.0
        inference_time = 0.0
        sample_send_time = 0.0
        num_large_model_step = 0
        prefill_time = 0
    dist.barrier()

    with torch.no_grad():
        while True:
            control_tensor = torch.tensor(0,device=DEVICE)
            dist.broadcast(control_tensor,draft_rank0)
            if control_tensor == 0:
                if args.Mode == "benchmark":
                    if prefill_time != 0:
                        print(f"Target: Receive and gather KV time: {receive_gather_time/num_large_model_step}; Inference time: {inference_time/num_large_model_step}; Sample and send result time: {sample_send_time/num_large_model_step}")
                        if prefill_time == 1:
                            receive_gather_time = 0.0
                            inference_time = 0.0
                            sample_send_time = 0.0
                            num_large_model_step = 0
                prefix = torch.zeros((BATCH_SIZE//2,128), device=DEVICE).long()
                dist.broadcast(prefix, draft_rank0)
                mini_batch_1_tree = PipeTree_Target(device=DEVICE, target_model_engine=target_model,prefix=prefix, temperature=args.T, top_p=args.P,
                                        max_length=MAX_LEN, grow_map = grow_map, batch_size=BATCH_SIZE//2, target_rank0=target_rank0, draft_rank0=draft_rank0, idx=0, max_target_seq=args.M)
                if args.Mode == "benchmark":
                    prefill_time += 1
            elif control_tensor == 1:
                prefix = torch.zeros((BATCH_SIZE//2,128), device=DEVICE).long()
                dist.broadcast(prefix, draft_rank0)
                mini_batch_2_tree = PipeTree_Target(device=DEVICE, target_model_engine=target_model,prefix=prefix, temperature=args.T, top_p=args.P,
                                        max_length=MAX_LEN, grow_map = grow_map, batch_size=BATCH_SIZE//2, target_rank0=target_rank0, draft_rank0=draft_rank0, idx=1, max_target_seq=args.M)
            elif control_tensor == 2:
                if args.Mode == "benchmark":
                    t1, t2, t3 = mini_batch_1_tree.verify(benchmark=True)
                    receive_gather_time+=t1
                    inference_time+=t2
                    sample_send_time+=t3
                    num_large_model_step+=1
                else:
                    mini_batch_1_tree.verify()
            elif control_tensor == 3:
                if args.Mode == "benchmark":
                    t1, t2, t3 = mini_batch_2_tree.verify(benchmark=True)
                    receive_gather_time+=t1
                    inference_time+=t2
                    sample_send_time+=t3
                    num_large_model_step+=1
                else:
                    mini_batch_2_tree.verify()
            elif control_tensor == 4:
                if args.Mode == "benchmark":
                    print(f"Target: Receive and gather KV time: {receive_gather_time/num_large_model_step}; Inference time: {inference_time/num_large_model_step}; Sample and send result time: {sample_send_time/num_large_model_step}")
                break

elif global_rank in args.draft_group:
    use_tp = len(args.draft_group)>1
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    if args.dataset == 'wiki':
        tokenized_dataset_eval = convert_wiki_dataset(tokenizer=tokenizer).select(list(range(args.start, args.end)))
    elif args.dataset == 'cnn':
        tokenized_dataset_eval = convert_cnn_dataset(tokenizer=tokenizer).select(list(range(args.start, args.end)))
    else:
        tokenized_dataset_eval = convert_c4_dataset(tokenizer=tokenizer,file_path="../dataset/c4_small.json").select(list(range(args.start, args.end)))
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    dataloader = DataLoader(tokenized_dataset_eval, batch_size=BATCH_SIZE, collate_fn=data_collator, shuffle=False, drop_last=True)

    path = args.growmap
    grow_map = torch.load(path)
    tree_size = grow_map["size"]
    idx_lists = grow_map["roots"]
    branch_lists = grow_map['branches']
    draft_step = len(grow_map["roots"])

    MAX_LEN = args.M + tree_size
    DRAFT_MODEL_CHECKPOINT = args.model
    DTYPE = torch.bfloat16

    sampling_callables = {}
    sample_gather_indices = {}
    for i in range(draft_step - 1):
        idx_len = len(idx_lists[i])
        num_samples = max(branch_lists[i])
        sampling_callables[i] = cuda_graph_for_sampling_argmax_batch(device=DEVICE,
            max_length=args.M, idx_len=idx_len, num_samples=num_samples,
            temperature=args.T, tree_size=tree_size, dtype = DTYPE, batch_size=BATCH_SIZE//2)  
    for i in range(draft_step - 1):
        ith_gather_list = []
        max_num_samples = max(branch_lists[i])
        for j, branch in enumerate(branch_lists[i]):
            branch_index = torch.arange(branch, device=DEVICE, dtype=torch.long)
            branch_index = branch_index + j * max_num_samples
            ith_gather_list.append(branch_index)
        ith_gather_list = torch.cat(ith_gather_list)
        sample_gather_indices[i] = ith_gather_list

    cg_list_draft = [sum(x) for x in branch_lists]
    cg_list_draft.append(1)
    target_rank0 = args.target_group[0]
    draft_rank0 = args.draft_group[0]

    draft_model = LMBackend(dtype=DTYPE, device=DEVICE, dec_list=cg_list_draft)
    draft_model.load_model(DRAFT_MODEL_CHECKPOINT, use_tp=use_tp, rank_group = args.draft_group, process_group = draft_group)
    if args.compile:
        draft_model.compile()
    draft_model.setup_caches(max_batch_size=BATCH_SIZE//2, max_seq_length=MAX_LEN, max_depth=draft_step)
    dist.barrier()
    dist.barrier()
    if args.Mode == "fast":
        simulation_fast(draft_model=draft_model, dataloader=dataloader, max_length=MAX_LEN, grow_map = grow_map, sampling_callables=sampling_callables, sample_gather_indices = sample_gather_indices, target_rank0 = target_rank0, draft_rank0 = draft_rank0)
    else:
        simulation_benchmark(draft_model=draft_model, dataloader=dataloader, max_length=MAX_LEN, grow_map = grow_map, sampling_callables=sampling_callables, sample_gather_indices = sample_gather_indices, target_rank0 = target_rank0, draft_rank0 = draft_rank0)

