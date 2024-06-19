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
from transformers import LlamaForCausalLM

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

causal_mask = torch.tril(torch.ones(max_seq_length, max_seq_length, dtype=torch.bool, device=device))
causal_mask[6][5] = False
print(causal_mask)

llm = LMBackend(dtype=precision, device=device)
llm.load_model(checkpoint_path, use_tp=False)
if args.compile:
    llm.compile()


llm.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)

prompt = torch.tensor([[    1, 15043, 29892,   590,  1024]], device=device,
       dtype=torch.int32)
input_pos = torch.tensor([0, 1, 2, 3, 4], device=device)
mask = causal_mask[:5]

dec =  torch.tensor([[518]], device=device, dtype=torch.int32)
dec_pos = torch.tensor([5], device=device, dtype=torch.int32)
cache_pos = torch.tensor([13], device=device, dtype=torch.int32)
dec_mask = causal_mask[4:5].clone()
dec_mask[0][13] = True
dec_mask = dec_mask[None, None, :, :]
print(dec_mask)

dec1 =  torch.tensor([[627]], device=device, dtype=torch.int32)
dec_pos1 = torch.tensor([5], device=device, dtype=torch.int32)
cache_pos1 = torch.tensor([14], device=device, dtype=torch.int32)
dec_mask1 = causal_mask[4:5].clone()
dec_mask1[0][14:15] = True
dec_mask1 = dec_mask1[None, None, :, :]
print(dec_mask1)

dec2 =  torch.tensor([[627]], device=device, dtype=torch.int32)
dec_pos2 = torch.tensor([6], device=device, dtype=torch.int32)
cache_pos2 = torch.tensor([15], device=device, dtype=torch.int32)
dec_mask2 = causal_mask[5:6].clone()
dec_mask2[0][15] = True
dec_mask2 = dec_mask2[None, None, :, :]
print(dec_mask2)

with torch.inference_mode():
        logits1 = llm.encode(input_ids=prompt, position_ids=input_pos, storage_ids=None, attention_mask=None)
        

        logits2 = llm.inference(input_ids=dec, position_ids=dec_pos, storage_ids=cache_pos, attention_mask=dec_mask)
        # logits = model_forward(model=model,x=dec,input_pos=dec_pos, cache_pos=cache_pos, attention_mask=dec_mask)
        

        logits3 = llm.inference(input_ids=dec1, position_ids=dec_pos1, storage_ids=cache_pos1, attention_mask=dec_mask1).clone()
        # logits = model_forward(model=model,x=dec1,input_pos=dec_pos1, cache_pos=cache_pos1, attention_mask=dec_mask1)
             
        llm.gather_kv_incremental(indices=[14], offset=5)

        logits4 = llm.inference(input_ids=dec2, position_ids=dec_pos2, storage_ids=cache_pos2, attention_mask=dec_mask2).clone()
        #logits = model_forward(model=model,x=dec1,input_pos=dec_pos1, cache_pos=cache_pos1, attention_mask=dec_mask1)
        
del llm
llm = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.bfloat16).to(device)
llm.eval()
input_ids = torch.LongTensor([
        [
     1, 15043, 29892,   590,  1024
        ]
    ]).to(device)
past_key_values = None
with torch.inference_mode():
    outputs = llm(input_ids,use_cache=True, past_key_values=past_key_values)
    _logits1 = outputs.logits
    past_key_values = outputs.past_key_values
    
    new_input_ids = torch.LongTensor([
        [
            627
        ]
    ]).to(device)
    
    outputs = llm(new_input_ids,use_cache=True, past_key_values=past_key_values)
    _logits2 = outputs.logits
    past_key_values = outputs.past_key_values
    
    new_input_ids = torch.LongTensor([
        [
            627
        ]
    ]).to(device)
    
    outputs = llm(new_input_ids,use_cache=True, past_key_values=past_key_values)
    _logits3 = outputs.logits

assert torch.allclose(logits1, _logits1, rtol=0.05, atol=0.05)
print(logits3)
print(_logits2)
assert torch.allclose(logits3, _logits2, rtol=0.05, atol=0.05)
print(logits4)
print(_logits3)
assert torch.allclose(logits4, _logits3, rtol=0.05, atol=0.05)

print("Success Verified")

        











