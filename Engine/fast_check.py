import itertools
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch._dynamo.config
import torch._inductor.config
from FastHesse.Engine.model import Transformer
import argparse
def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif ("cpu" in device) or ("mps" in device):
        pass
    else:
        print(f"device={device} is not yet suppported")

def _load_model(checkpoint_path, device, precision):

    with torch.device('meta'):
        model = Transformer.from_name(checkpoint_path.parent.name)

    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    if "model" in checkpoint and "stories" in str(checkpoint_path):
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint, assign=True)

    model = model.to(device=device, dtype=precision)
    return model.eval()

def model_forward(model :Transformer, x :torch.Tensor, input_pos :torch.Tensor, cache_pos :torch.Tensor = None, attention_mask: torch.Tensor = None) -> torch.Tensor:
    return model(x, input_pos, cache_pos, attention_mask)

def prefill(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, cache_pos :torch.Tensor = None, attention_mask: torch.Tensor = None) -> torch.Tensor:
    # input_pos: [B, S]
    logits = model(x, input_pos, cache_pos, attention_mask)
    return logits


torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

parser = argparse.ArgumentParser(description='Your CLI description.')

parser.add_argument('--checkpoint_path', type=Path, default=Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"), help='Model checkpoint path.')
parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')
parser.add_argument('--batch', type=int, help='batch size')
parser.add_argument('--maxlen', type=int, help='max len')
parser.add_argument('--declen', type=int, help='decode len')
parser.add_argument('--prefixlen', type=int, help='prefix len')
parser.add_argument('--device', type=str, help='device')
args = parser.parse_args()

checkpoint_path = args.checkpoint_path
device = args.device
precision = torch.bfloat16
use_tp = False
max_seq_length = args.maxlen
max_batch_size = args.batch
prefix_len = args.prefixlen
declen = args.declen
warm_up = 10
T = 500

causal_mask = torch.tril(torch.ones(max_seq_length, max_seq_length, dtype=torch.bool, device='cuda:9'))

model = _load_model(checkpoint_path, device, precision)
if args.compile:
    model_forward = torch.compile(model_forward, mode="reduce-overhead", fullgraph=True)

with torch.device(device):
        model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)


prompt = torch.tensor([[    1, 15043, 29892,   590,  1024,   338]], device='cuda:9',
       dtype=torch.int32)
input_pos = torch.tensor([0, 1, 2, 3, 4, 5], device='cuda:9')
mask = causal_mask[:6]

dec =  torch.tensor([[518]], device='cuda:9', dtype=torch.int32)
dec_pos = torch.tensor([6], device='cuda:9', dtype=torch.int32)
cache_pos = torch.tensor([6], device='cuda:9', dtype=torch.int32)
dec_mask = causal_mask[6:7][None, None, :, :]

dec1 =  torch.tensor([[627]], device='cuda:9', dtype=torch.int32)
dec_pos1 = torch.tensor([7], device='cuda:9', dtype=torch.int32)
cache_pos1 = torch.tensor([7], device='cuda:9', dtype=torch.int32)
dec_mask1 = causal_mask[7:8][None, None, :, :]

with torch.inference_mode():
        logits = prefill(model, prompt.view(1, -1), input_pos)
        print(logits)
        logits = model_forward(model=model,x=dec,input_pos=dec_pos, cache_pos=cache_pos, attention_mask=dec_mask)
        print(logits)
        logits = model_forward(model=model,x=dec1,input_pos=dec_pos1, cache_pos=cache_pos1, attention_mask=dec_mask1)
        print(logits)
        











