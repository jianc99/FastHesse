import sys
import time
from pathlib import Path
import sys
sys.path.append("..")
import torch
import torch._dynamo.config
import torch._inductor.config
import argparse
from FastHesse.New_Engine.backend import LMBackend

parser = argparse.ArgumentParser(description='Your CLI description.')
parser.add_argument('--checkpoint_path', type=Path, default=Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"), help='Model checkpoint path.')
parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')
parser.add_argument('--batch', type=int, help='batch size')
parser.add_argument('--maxlen', type=int, help='max len')
parser.add_argument('--prefixlen', type=int, help='prefix len')
parser.add_argument('--declen_list', nargs='+', type=int, help='Group of dec len')
parser.add_argument('--rank_group', nargs='+', type=int, help='Group of ranks')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
global print
from FastHesse.New_Engine.tp import init_dist
rank = init_dist()
use_tp = len(args.rank_group)>1
if use_tp:
    if rank != 0:
        # only print on rank 0
        print = lambda *args, **kwargs: None
print(f"Using device={device}")

checkpoint_path = args.checkpoint_path
precision = torch.bfloat16
max_seq_length = args.maxlen
max_batch_size = args.batch
prefix_len = args.prefixlen
dec_list = args.declen_list

warm_up = 10
T = 1000

causal_mask = torch.tril(torch.ones(max_seq_length, max_seq_length, dtype=torch.bool, device=device))

llm = LMBackend(dtype=precision, device=device, dec_list=dec_list)
llm.load_model(checkpoint_path, use_tp=use_tp, rank_group=args.rank_group)
if args.compile:
    llm.compile()
llm.setup_caches(max_batch_size=max_batch_size, max_seq_length=max_seq_length)

prefill_storage_ids = torch.arange(0, prefix_len, device=device)
prefill_attention_mask = causal_mask[:prefix_len][None, None, :, :].repeat(max_batch_size,1,1,1)
prompt = torch.randint(low=3, high=30000, size=(max_batch_size, prefix_len), device=device)
prefill_pos = torch.arange(0, prefix_len, device=device).repeat(max_batch_size,1)
llm.encode(input_ids=prompt, position_ids=prefill_pos, storage_ids=prefill_storage_ids, attention_mask=prefill_attention_mask)

# indices = torch.tensor([16,17,19], device=device).long().repeat(max_batch_size,1)
# offset = torch.tensor(4, device=device).long().repeat(max_batch_size,1)
# k = torch.arange(0, 32, device=device).repeat(max_batch_size,1)*2
# dest_indices = offset + torch.arange(0, indices.size(1), device=k.device).repeat(max_batch_size,1)
# k[:,dest_indices] = k[:,indices]
# print(k)
# print(indices)
# time.sleep(1000)
for _ in range(3):
    for declen in dec_list:
        dec = torch.randint(low=3, high=30000, size=(max_batch_size, declen), device=device)
        dec_pos = torch.arange(prefix_len, prefix_len + declen, device=device).unsqueeze(0).repeat(max_batch_size,1)
        cache_pos = torch.arange(prefix_len, prefix_len + declen, device=device)
        dec_mask = causal_mask[prefix_len:prefix_len + declen][None, None, :, :].repeat(max_batch_size,1,1,1)

        with torch.inference_mode():
                for _ in range(warm_up):
                    logits = llm.inference(input_ids=dec, position_ids=dec_pos, storage_ids=cache_pos, attention_mask=dec_mask)
                    # llm.gather_kv_incremental(indices, offset)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                for _ in range(T):
                    logits = llm.inference(input_ids=dec, position_ids=dec_pos, storage_ids=cache_pos, attention_mask=dec_mask)
                torch.cuda.synchronize()
                t2 = time.perf_counter()
                # for _ in range(T):
                #     # llm.gather_kv_incremental(indices, offset)
                torch.cuda.synchronize()
                t3 = time.perf_counter()

        print("Max Length :{}, Decode Length :{}, Prefix Length :{}, inference time:{}s, gather kv time:{}s".format(max_seq_length, declen, prefix_len, (t2 - t1)/ T, (t3-t2)/T))

