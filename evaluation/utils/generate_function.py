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
        edit_token_mask = confidence_unmask[j] > 0
        if enable_editing:
            mask = edit_token_mask | mask_token_mask
        else:
            mask = mask_token_mask
        transfer_index[j] = mask
    block_x[transfer_index] = x0[transfer_index]
    return block_x,transfer_index.sum()/confidence.shape[0]


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
    model_path = "/path/to/dLLM-Var/
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
    prompt_1 = "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
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