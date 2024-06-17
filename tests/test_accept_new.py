import sys
sys.path.append("..")
import argparse
import time
from pathlib import Path
import torch
import torch.distributed as dist
from transformers import DataCollatorForLanguageModeling, AutoTokenizer
from FastHesse.NewTree.BatchTree import BatchSTreeTest
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from FastHesse.Data.data_converter import convert_wiki_dataset, convert_cnn_dataset, convert_c4_dataset
from FastHesse.NewTree.utils import cuda_graph_for_sampling_argmax
from FastHesse.New_Engine.utlis import setup_seed
from FastHesse.New_Engine.backend import LMBackend

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=Path, default=Path("/home/jianc2/gpt-fast/checkpoints/meta-llama/Llama-2-7b-hf/model.pth"), help='model')
parser.add_argument('--target', type=Path, default=Path("/home/jianc2/gpt-fast/checkpoints/meta-llama/Llama-2-70b-hf/model.pth"), help='target model')
parser.add_argument('--dataset', type=str, default="cnn", help='dataset path')
parser.add_argument('--start', type=int, default=0, help='start')
parser.add_argument('--end', type=int, default=200, help='end')
parser.add_argument('--T', type=float, default=0.6, help='temperature')
parser.add_argument('--P', type=float, default=0.9, help='top_p')
parser.add_argument('--M', type=int, default=256, help='max length')
parser.add_argument('--W', type=int, default=32, help='max width')
parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument('--dst', type=str, default="btree_acc.pt", help='destination for accepetance rate vector')
# Target model information
parser.add_argument('--target_group', nargs='+', type=int, help='Target group of ranks')
# Draft model information
parser.add_argument('--draft_group', nargs='+', type=int, help='Target group of ranks')
parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')
args = parser.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
global print
from FastHesse.New_Engine.tp import init_dist
use_tp = True
if use_tp:
    global_rank = init_dist()
if use_tp:
    if global_rank != 0:
        # only print on rank 0
        print = lambda *args, **kwargs: None
setup_seed(args.seed)
BATCH_SIZE = 1

def simulation_fast(target_model : LMBackend, draft_model: LMBackend, dataloader: DataLoader, T=0.6, top_p=0.9, 
            max_length=512, grow_map=None, sampling_callables = None,
            sample_gather_indices = None, max_width=32):
    num_eval_steps = len(dataloader)
    num_decoding_steps = 0
    num_large_model_steps = 0
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=False)
    accept_list = torch.zeros(max_width+1).to(DEVICE)
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader), total=num_eval_steps):
            input_ids = batch['input_ids'][..., :128]
            labels = batch['labels'][..., :128]
            terminate = False
            if (labels[:, -1] == -100)._is_any_true(): terminate = True
            spectree = BatchSTreeTest (prefix=input_ids, device=DEVICE, temperature=T, top_p=top_p,
                                    draft_model_engine=draft_model, target_model_engine=target_model, max_length=max_length, grow_map=grow_map,
                                   sampling_callables=sampling_callables, sample_gather_indices = sample_gather_indices, batch_size=BATCH_SIZE, max_target_seq=256
                                    )
            longest=128
            while longest < 256 and terminate == False:
                spectree.construct_grow_map()
                num_nodes, terminate, accept_pos = spectree.verify()
                longest = num_nodes.max()
                num_large_model_steps += 1
                if accept_pos!=-1:
                    accept_list[accept_pos]+=1
                else:
                    accept_list[0]+=1
            draft_model.clear_kv()
            target_model.clear_kv()
            num_decoding_steps += (num_nodes.sum() - input_ids.shape[1]*BATCH_SIZE)
            if dist.get_rank() == 0:
                for i in range(BATCH_SIZE):
                    print(tokenizer.decode(spectree.tokens[i,:num_nodes[i]]))
                print("total decoding steps: {}".format(num_decoding_steps), "large model steps: {}".format(num_large_model_steps), "avg decoding step: {}".format(num_decoding_steps / num_large_model_steps))
                print(accept_list)
    if dist.get_rank() == 0:
        print("total decoding steps: {}".format(num_decoding_steps), "large model steps: {}".format(num_large_model_steps), "avg decoding step: {}".format(num_decoding_steps / num_large_model_steps))
        accept_list = accept_list / accept_list.sum() 
        print(accept_list)
        torch.save(accept_list, args.dst)
    return num_decoding_steps / num_large_model_steps

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

mask = torch.full((1+args.W, 1+args.W), 0, dtype=torch.bool)
mask[:,0]= True
depth=[0]
successors = [list(range(1, args.W+1))]
for i in range(1,args.W+1):
    mask[i,i]= True
    depth.append(1)
    successors.append([])
grow_map={
    "size": args.W+1,
    "roots": [[0], list(range(1, args.W+1))],
    "branches": [[args.W]],
    "Successors": successors,
    "mask": mask,
    "depth": torch.tensor(depth)
}

tree_size = grow_map["size"]
idx_lists = grow_map["roots"]
branch_lists = grow_map['branches']
draft_step = len(grow_map["roots"])

MAX_LEN = args.M + tree_size
TARGET_MODEL_CHECKPOINT = args.target
DRAFT_MODEL_CHECKPOINT = args.model
DTYPE = torch.bfloat16

sampling_callables = {}
sample_gather_indices = {}
for i in range(draft_step - 1):
    idx_len = len(idx_lists[i])
    num_samples = max(branch_lists[i])
    sampling_callables[i] = cuda_graph_for_sampling_argmax(device=DEVICE,
        max_length=args.M, idx_len=idx_len, num_samples=num_samples,
        temperature=args.T, tree_size=tree_size)  
for i in range(draft_step - 1):
    ith_gather_list = []
    max_num_samples = max(branch_lists[i])
    for j, branch in enumerate(branch_lists[i]):
        branch_index = torch.arange(branch, device=DEVICE, dtype=torch.long)
        branch_index = branch_index + j * max_num_samples
        ith_gather_list.append(branch_index)
    ith_gather_list = torch.cat(ith_gather_list)
    sample_gather_indices[i] = ith_gather_list

dec_list_target = [tree_size]
dec_list_draft = [sum(x) for x in branch_lists]
dec_list_draft.append(1)

draft_model = LMBackend(dtype=DTYPE, device=DEVICE, dec_list=dec_list_draft)
draft_model.load_model(DRAFT_MODEL_CHECKPOINT, use_tp=use_tp, rank_group = args.draft_group)
if args.compile:
    draft_model.compile()
draft_model.setup_caches(max_batch_size=BATCH_SIZE, max_seq_length=MAX_LEN)

target_model = LMBackend(dtype=DTYPE, device=DEVICE, dec_list=dec_list_target)
target_model.load_model(TARGET_MODEL_CHECKPOINT, use_tp=use_tp, rank_group = args.target_group)
if args.compile:
    target_model.compile()
target_model.setup_caches(max_batch_size=BATCH_SIZE, max_seq_length=MAX_LEN)

simulation_fast(target_model=target_model, draft_model=draft_model, dataloader=dataloader, T=args.T, top_p=args.P,
                                    max_length=MAX_LEN, grow_map = grow_map, sampling_callables=sampling_callables, sample_gather_indices = sample_gather_indices, max_width=args.W)

