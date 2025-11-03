import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "maomaocun/dLLM-Var"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda")

# 使用对话模板
messages = [
    {"role": "user", "content": "Can you tell me an engaging short story about a brave young astronaut who discovers an ancient alien civilization on a distant planet? Make it adventurous and heartwarming, with a twist at the end."}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
input_ids = inputs['input_ids']
attention_mask = inputs.get('attention_mask', torch.ones_like(input_ids))
result = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_gen_length=1024,
    block_length=64,
    threshold=0.9,
    eos_token_id=126348 
)
text = tokenizer.batch_decode(result, skip_special_tokens=True)
print(messages[0]['content']+"\n"+text[0])

