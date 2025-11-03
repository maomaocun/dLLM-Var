import time
from datetime import datetime
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
import numpy as np
import argparse

def update_kv_cache(old_key_values, block_length):
    outupt_past_key_values = []
    for i in range(len(old_key_values)):
        k,v = old_key_values[i]
        new_k,new_v = k[:,:,:-block_length,:],v[:,:,:-block_length,:]
        outupt_past_key_values.append((new_k,new_v))
        del k,v
    outupt_past_key_values = tuple(outupt_past_key_values)
    return outupt_past_key_values

def prefill_phase(model, input_ids,block_length):
    """Prefill phase: Process initial prompt and generate KV cache."""
    cache_position = torch.arange(input_ids.size(-1), device=input_ids.device)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            use_cache=True,
            cache_position=cache_position,
            return_dict=True
        )
    outupt_past_key_values = []
    for i in range(len(outputs.past_key_values)):
        k,v = outputs.past_key_values[i]
        new_k,new_v = k[:,:,:-block_length,:],v[:,:,:-block_length,:]
        outupt_past_key_values.append((new_k,new_v))
        del k,v
    outupt_past_key_values = tuple(outupt_past_key_values)
    return {
        'input_ids': input_ids,
        'logits': outputs.logits[:,-block_length:,:],
        'past_key_values': outupt_past_key_values,
        'cache_position': cache_position
    }



# def apply_delete_insert(block_x, delete_id, insert_id, mask_id):
#     device = block_x.device
#     B, L = block_x.shape
#     result = torch.zeros_like(block_x)
#     for b in range(B):
#         seq = block_x[b]
#         # Delete operation: process from back to front
#         delete_pos = (seq == delete_id).nonzero(as_tuple=True)[0].flip(0)  # reverse to process from end
#         for pos in delete_pos:
#             pos = pos.item()
#             seq = torch.cat([seq[:pos], seq[pos+1:], torch.tensor([mask_id], device=device)])
        
#         # Insert operation: process from back to front
#         insert_pos = (seq == insert_id).nonzero(as_tuple=True)[0].flip(0)  # reverse
#         for pos in insert_pos:
#             pos = pos.item()
#             seq = torch.cat([seq[:pos], torch.tensor([mask_id, mask_id], device=device), seq[pos+1:]])
#             seq = seq[:-1]  # delete last token
        
#         result[b] = seq
#     return result

# def apply_delete_insert(block_x, block_x_old, delete_id, insert_id, mask_id):
#     device = block_x.device
#     B, L = block_x.shape
#     result = torch.zeros_like(block_x)
#     for b in range(B):
#         seq = block_x[b]
#         seq_x0 = block_x_old[b]
#         # Delete operation: process from back to front
#         delete_pos = (seq == delete_id).nonzero(as_tuple=True)[0].flip(0)  # reverse to process from end

#         for pos in delete_pos:
#             pos = pos.item()
#             before = seq.clone()
#             seq = torch.cat([seq[:pos], seq[pos+1:], torch.tensor([mask_id], device=device)])
#             seq_x0 = torch.cat([seq_x0[:pos], seq_x0[pos+1:], torch.tensor([mask_id], device=device)])
        
#         # Insert operation: process from back to front
#         insert_pos = (seq == insert_id).nonzero(as_tuple=True)[0].flip(0)  # reverse

#         for pos in insert_pos:
#             pos = pos.item()
#             before = seq.clone()
#             # print("seq vs seq_x0:", seq[pos], seq_x0[pos])
#             seq = torch.cat([seq[:pos], torch.tensor([seq_x0[pos], mask_id], device=device), seq[pos+1:]])
#             seq_x0 = torch.cat([seq_x0[:pos+1], torch.tensor([mask_id], device=device), seq_x0[pos+1:]])
#             seq = seq[:-1]  # delete last token
#             seq_x0 = seq_x0[:-1]  # keep seq_x0 same length
        
#         result[b] = seq
#     return result


def get_enclosed_mask(block_x, mask_id):
    """
    获取 batch 中每个序列被非 mask token 包围的 mask bool 矩阵。
    
    Args:
        block_x (torch.Tensor): 形状 [batch_size, seq_len] 的 token ID tensor。
        mask_id (int): mask token 的 ID 值。
    
    Returns:
        torch.Tensor: 形状 [batch_size, seq_len] 的 bool 矩阵，True 表示 enclosed mask 位置。
    """
    batch_size, seq_len = block_x.shape
    bool_matrix = torch.zeros(batch_size, seq_len, dtype=torch.bool,device=block_x.device)
    for b in range(batch_size):
        mask = (block_x[b] == mask_id)
        n = seq_len
        i = 0
        while i < n:
            if mask[i]:
                start = i
                while i < n and mask[i]:
                    i += 1
                end = i  # end 为 exclusive
                # 检查是否被非 mask 包围：start > 0 且 end < n
                if start > 0 and end < n:
                    bool_matrix[b, start:end] = True
                    # break
            else:
                i += 1
    return bool_matrix

def unmask_function(logits,block_x,mask_id,delete_id=None,insert_id=None,remasking='low_confidence',temperature=0.0,threshold=0.9,
top_p=0.95,top_k=10,enable_editing=False,token_per_steps=4):
    assert delete_id is not None and insert_id is not None
    mask_index = block_x == mask_id
    if remasking == 'low_confidence':
        x0 = torch.argmax(logits, dim=-1)
        p = F.softmax(logits, dim=-1)
        confidence = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
    else:
        raise NotImplementedError(remasking)
    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    confidence_mask = torch.where(mask_index, confidence, -np.inf)
    confidence_unmask = torch.where(~mask_index, confidence, -np.inf)
    for j in range(confidence.shape[0]):
        mask_token_mask = confidence_mask[j] > threshold
        if mask_token_mask.sum()==0 and mask_index.sum()!=0:
            max_conf_idx = torch.argmax(confidence_mask[j])
            mask_token_mask[max_conf_idx] = True
        # if mask_token_mask.sum() == 0 and mask_index[j].sum() != 0:
        #     num_select = min(token_per_steps, int(mask_index[j].sum()))
        #     _, top_indices = torch.topk(confidence_mask[j], num_select, dim=0)
        #     mask_token_mask[top_indices] = True
        edit_token_mask = confidence_unmask[j] > 0
        if enable_editing:
            mask = edit_token_mask | mask_token_mask
        else:
            mask = mask_token_mask
        transfer_index[j] = mask
    block_x[transfer_index] = x0[transfer_index]
    return block_x,transfer_index.sum()/confidence.shape[0]

# 156895 llada moe <|mask|>
# 126336 llada dense <|mask|>

# 156892 llada moe <|endoftext|>
# 126081 llada dense <|endoftext|>


# 156900 llada moe <|role_end|>
# 126348 llada dense <|eot_id|>

@torch.no_grad()
def generate(model, attention_mask, input_ids, gen_length=1024, block_length=128, temperature=0.0,
             remasking='low_confidence', mask_id=126336, delete_id=126084, insert_id=126085, eos_token=126081, threshold=0.6, top_p=0.95, top_k=10, enable_editing=False):
    batchsize, prompt_length = input_ids.shape
    max_num_blocks = gen_length // block_length
    output_ids = input_ids
    inference_times = 0
    tokens = 0
    outputs = None
    block_x = torch.full((batchsize, block_length), mask_id, dtype=torch.long).to(model.device)
    output_ids = torch.cat([output_ids, block_x], dim=-1)
    prefill_outputs = prefill_phase(model, output_ids, block_length)
    past_key_values = prefill_outputs['past_key_values']
    logits = prefill_outputs['logits']
    output_ids[:,-block_length:], token_per_inference = unmask_function(
        logits=logits, block_x=output_ids[:,-block_length:], mask_id=mask_id, delete_id=delete_id, insert_id=insert_id, remasking=remasking,
        temperature=temperature, threshold=threshold,top_p=top_p,top_k=top_k,enable_editing=enable_editing
    )
    tokens += token_per_inference
    
    for j in range(max_num_blocks):
        max_inference_steps_per_block = block_length
        inference_step_per_block = 0
        # threshold = max(0.6, threshold - 0.1 * j)  # 动态调整阈值，逐渐降低
        while (output_ids[:, prompt_length:] == mask_id).sum():
            inference_step_per_block += 1
            inference_times += 1
            outputs = model(
                input_ids=output_ids[:,-block_length:],
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )
            output_ids[:,-block_length:], token_per_inference = unmask_function(
                logits=outputs.logits, block_x=output_ids[:,-block_length:], mask_id=mask_id, delete_id=delete_id, insert_id=insert_id, remasking=remasking,
                temperature=temperature, threshold=threshold,top_p=top_p,top_k=top_k,enable_editing=enable_editing
            )
            tokens += token_per_inference
            if inference_step_per_block >= max_inference_steps_per_block:
                print(f"blocks{j},达到单block最大推理步数")
                break

        if (output_ids[:, prompt_length:] == eos_token).any() or (output_ids[:, prompt_length:] == 126348).any():
            print(f"blocks{j},EOS token generated")
            tokens_per_infer = tokens / inference_times if inference_times > 0 else 0
            return output_ids[:, prompt_length:],float(tokens_per_infer)

        if j != max_num_blocks - 1:
            block_x = torch.full((batchsize, block_length), mask_id, dtype=torch.long).to(model.device)
            output_ids = torch.cat([output_ids, block_x], dim=-1)
            inference_times += 1
            outputs = model(
                input_ids=output_ids[:,-block_length*2:],
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )
            output_ids[:,-block_length*2:], token_per_inference = unmask_function(
                logits=outputs.logits, block_x=output_ids[:,-block_length*2:], mask_id=mask_id, delete_id=delete_id, insert_id=insert_id, remasking=remasking,
                temperature=temperature, threshold=threshold,top_p=top_p,top_k=top_k,enable_editing=enable_editing
            )
            tokens += token_per_inference
            if outputs is not None:
                past_key_values = update_kv_cache(outputs.past_key_values,block_length)


    print("已达到最大生成长度")
    tokens_per_infer = tokens / inference_times if inference_times > 0 else 0
    return output_ids[:, prompt_length:],float(tokens_per_infer)


if __name__ == "__main__":
    # model_path = "/mnt/innovator/model/yangyicun/LLaDA-8B-Base-inference-fp8"
    # model_path = "/mnt/public/yangyicun/pretrained_model/LLaDA-8B-Instruct"
    # model_path = "/mnt/innovator/code/yangyicun/checkpoint/always_eos_loss_only_mask/checkpoint-12000"
    # model_path = "/mnt/innovator/model/yangyicun/LLaDA-MoE-7B-A1B-Base"
    # model_path = "/mnt/innovator/model/yangyicun/LLaDA-MoE-7B-A1B-Base-fusemoe"
    model_path = "/mnt/innovator/code/yangyicun/checkpoint/block_mask"
    # model_path = "/mnt/innovator/code/yangyicun/training_code/dLLM-Factory-main/sft-pretrain-v3-fp8-moe/sft_save/llada-moe-only-mask-8k-test/checkpoint-20000"
    # model_path = "/mnt/innovator/code/yangyicun/training_code/dLLM-Factory-main/sft-pretrain-v3-fp8-moe/sft_save/llada-moe-only-mask-8k-test/checkpoint-2000"
    # model_path = "/mnt/public/yangyicun/test/dLLM-Factory-main/sft_pretrain-v2/sft_save/llada-sft-global-batch-size=128/checkpoint-16000"
    # model_path = "/mnt/public/yangyicun/checkpoint/llada_sft/8_15/save"
    # model_path = "/mnt/public/yangyicun/pretrained_model/Dream-7B-Instruct"
    # model_path = "/mnt/public/yangyicun/pretrained_model/Dream-7B-Instruct"
    # model_path = "/cpfs02/shared/llmit6/liudawei/maomaocun/models/LLaDA-1.5"
    print(f"当前加载的模型路径为: {model_path}")
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    # tokenizer.pad_token = tokenizer.bos_token
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=None,
        trust_remote_code=True
    ).to(device)
    # prompt_1 = "Question: Jen and Tyler are gymnasts practicing flips. Jen is practicing the triple-flip while Tyler is practicing the double-flip. Jen did sixteen triple-flips during practice. Tyler flipped in the air half the number of times Jen did. How many double-flips did Tyler do?\nAnswer: Jen did 16 triple-flips, so she did 16 * 3 = <<16*3=48>>48 flips.\nTyler did half the number of flips, so he did 48 / 2 = <<48/2=24>>24 flips.\nA double flip has two flips, so Tyler did 24 / 2 = <<24/2=12>>12 double-flips.\n#### 12\n\nQuestion: Four people in a law firm are planning a party. Mary will buy a platter of pasta for $20 and a loaf of bread for $2. Elle and Andrea will split the cost for buying 4 cans of soda which cost $1.50 each, and chicken wings for $10. Joe will buy a cake that costs $5. How much more will Mary spend than the rest of the firm put together?\nAnswer: Mary will spend $20 + $2 = $<<20+2=22>>22.\nElle and Andrea will spend $1.5 x 4 = $<<1.5*4=6>>6 for the soda.\nElle and Andrea will spend $6 + $10 = $<<6+10=16>>16 for the soda and chicken wings.\nElle, Andrea, and Joe together will spend $16 + $5 = $<<16+5=21>>21.\nSo, Mary will spend $22 - $21 = $<<22-21=1>>1 more than all of them combined.\n#### 1\n\nQuestion: A charcoal grill burns fifteen coals to ash every twenty minutes of grilling. The grill ran for long enough to burn three bags of coals. Each bag of coal contains 60 coals. How long did the grill run?\nAnswer: The grill burned 3 * 60 = <<3*60=180>>180 coals.\nIt takes 20 minutes to burn 15 coals, so the grill ran for 180 / 15 * 20 = <<180/15*20=240>>240 minutes.\n#### 240\n\nQuestion: A bear is preparing to hibernate for the winter and needs to gain 1000 pounds. At the end of summer, the bear feasts on berries and small woodland animals. During autumn, it devours acorns and salmon. It gained a fifth of the weight it needed from berries during summer, and during autumn, it gained twice that amount from acorns. Salmon made up half of the remaining weight it had needed to gain. How many pounds did it gain eating small animals?\nAnswer: The bear gained 1 / 5 * 1000 = <<1/5*1000=200>>200 pounds from berries.\nIt gained 2 * 200 = <<2*200=400>>400 pounds from acorns.\nIt still needed 1000 - 200 - 400 = <<1000-200-400=400>>400 pounds.\nThus, it gained 400 / 2 = <<400/2=200>>200 pounds from salmon.\nTherefore, the bear gained 400 - 200 = <<400-200=200>>200 pounds from small animals.\n#### 200\n\nQuestion: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?\nAnswer:"
    # prompt_1 = "Jen and Tyler are gymnasts practicing flips. Jen is practicing the triple-flip while Tyler is practicing the double-flip. Jen did sixteen triple-flips during practice. Tyler flipped in the air half the number of times Jen did. How many double-flips did Tyler do?<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>"
    prompt_1 = "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    # prompt_1 = "Jen and Tyler are gymnasts practicing flips. Jen is practicing the triple-flip while Tyler is practicing the double-flip. Jen did sixteen triple-flips during practice. Tyler flipped in the air half the number of times Jen did. How many double-flips did Tyler do?"
    # prompt_1 = "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    # prompt_1 = "Write a short story (300-500 words) set in a futuristic city where AI governs daily life. The protagonist is a young rebel who discovers a hidden flaw in the AI system that could either save or destroy the city. Include vivid descriptions of the setting and explore the protagonist's internal conflict about whether to expose the flaw. Use a third-person perspective and a suspenseful tone."
    # input_ids = tokenizer.encode(prompt,return_tensors="pt").to(device)
    # input_ids = tokenizer([prompt_1, prompt_2], return_tensors="pt",  padding=True).input_ids.to(device)
    # attention_mask = tokenizer([prompt_1, prompt_2], return_tensors="pt",  padding=True).attention_mask.to(device)
    def apply_chat_template_to_prompts(prompts):
        formatted_prompts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            try:
                formatted = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True  # 添加生成提示，如 </s> 或类似
                )
                formatted_prompts.append(formatted)
            except AttributeError:
                # Fallback: 如果无 chat template，直接使用原始 prompt
                print("警告: tokenizer 无 chat template，使用原始 prompt")
                formatted_prompts.append(prompt)
        return formatted_prompts
    # prompt_1 = apply_chat_template_to_prompts([prompt_1])
    prompt_1 = [prompt_1]
    print(prompt_1)
    input_ids = tokenizer(prompt_1, return_tensors="pt",  padding=True).input_ids.to(device)
    attention_mask = tokenizer(prompt_1, return_tensors="pt",  padding=True).attention_mask.to(device)
    _,_ = generate(
        model, attention_mask, input_ids, delete_id=126084, insert_id=126085
    )
    # 创建 CUDA 事件
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    # 记录开始时间
    start_time.record()
    # 执行生成函数
    result_ids,tokens_per_infer = generate(
        model, attention_mask, input_ids, delete_id=126084, insert_id=126085
    )

    # 记录结束时间
    end_time.record()

    # 同步 CUDA 操作，确保时间戳准确
    torch.cuda.synchronize()

    # 计算时间差（以毫秒为单位）
    elapsed_time = start_time.elapsed_time(end_time)

    # print(result_ids.shape)
    print(f"总耗时: {elapsed_time/1000:.2f} 秒")
    print(f"tokens/inference_times: {tokens_per_infer:.4f}")
    print(f"tokens per second: {result_ids.shape[1]/(elapsed_time/1000):.4f}")
    # result_text = tokenizer.batch_decode(result_ids, skip_special_tokens=True)
    for i in range(result_ids.shape[0]):
        result_text_i = tokenizer.decode(result_ids[i], skip_special_tokens=False)
        print(f"\n最终输出{i}: {result_text_i}")