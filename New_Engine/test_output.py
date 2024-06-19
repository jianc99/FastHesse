import itertools
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch._dynamo.config
import torch._inductor.config
from FastHesse.New_Engine.model import Transformer
import argparse
from backend import LMBackend
from transformers import LlamaForCausalLM, LlamaTokenizer

parser = argparse.ArgumentParser(description='Your CLI description.')
parser.add_argument('--checkpoint_path', type=Path, default=Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"), help='Model checkpoint path.')
parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')
parser.add_argument('--batch', type=int, help='batch size')
parser.add_argument('--maxlen', type=int, help='max len')
parser.add_argument('--declen', type=int, help='decode len')
parser.add_argument('--prefixlen', type=int, help='prefix len')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
global print
from tp import maybe_init_dist
rank = maybe_init_dist()
use_tp = rank is not None
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
declen = args.declen

warm_up = 10
T = 500

causal_mask = torch.tril(torch.ones(max_seq_length, max_seq_length, dtype=torch.bool, device=device))

llm = LMBackend(dtype=precision, device=device)
llm.load_model(checkpoint_path, use_tp=use_tp)
if args.compile:
    llm.compile()


llm.setup_caches(max_batch_size=max_batch_size, max_seq_length=max_seq_length)