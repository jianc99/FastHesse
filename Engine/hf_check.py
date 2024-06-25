from transformers import LlamaForCausalLM
import torch
llm = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float32).cuda()


input_ids = torch.LongTensor([
        [
     1, 15043, 29892,   590,  1024
        ]
    ]).cuda()
past_key_values = None
with torch.inference_mode():
    outputs = llm(input_ids,use_cache=True, past_key_values=past_key_values)
    print(outputs.logits)
    past_key_values = outputs.past_key_values
    
    new_input_ids = torch.LongTensor([
        [
            627
        ]
    ]).cuda()
    
    outputs = llm(new_input_ids,use_cache=True, past_key_values=past_key_values)
    print(outputs.logits)
    past_key_values = outputs.past_key_values
    
    new_input_ids = torch.LongTensor([
        [
            627
        ]
    ]).cuda()
    
    outputs = llm(new_input_ids,use_cache=True, past_key_values=past_key_values)
    print(outputs.logits)
    
    
    
