import json
import os
from pathlib import Path
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import argparse
from functools import partial

def tokenize_batch(batch, tokenizer, max_length=8192):
    """
    datasets.map 使用的分词函数，对一批数据进行处理。
    Args:
        batch (dict): a batch of examples from the dataset.
        tokenizer: The tokenizer instance.
        max_length (int): The maximum sequence length for the tokenizer.
    Returns:
        dict: A dictionary containing the tokenized 'input_ids', 'prompt_length', and 'token_count'.
    """
    tokenized_batch = {
        "input_ids": [],
        "prompt_length": [],
        "token_count": []  # 临时添加一列用于统计
    }
    # batch 是一个字典，例如 {"instruction": [...], "input": [...], "output": [...]}
    # 我们需要按行遍历
    for i in range(len(batch["instruction"])):
        # 确保所有必需的字段都存在且不为 None
        instruction = batch["instruction"][i] or ""
        input_text = batch["input"][i] or ""
        output_text = batch["output"][i] or ""
        # 拼接文本字段
        text = f"{instruction}{input_text}{output_text}<|endoftext|>"
        prompt = f"{instruction}{input_text}"
        # 对完整文本进行分词
        tokens = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors=None
        )
        if len(tokens["input_ids"]) >= max_length:
            continue
        # 对 prompt 进行分词以获取其长度
        prompt_tokens = tokenizer(
            prompt,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors=None
        )
        tokenized_batch["input_ids"].append(tokens["input_ids"])
        tokenized_batch["prompt_length"].append(len(prompt_tokens["input_ids"]))
        tokenized_batch["token_count"].append(len(tokens["input_ids"]))
    return tokenized_batch

def process_files_with_datasets(input_dir, output_file, tokenizer, max_length=8192, num_workers=None):
    """
    使用 'datasets' 库并行处理目录中的所有 JSONL 文件。
    Args:
        input_dir (str): 输入文件夹的路径。
        output_file (str): 输出的单个 tokenized JSONL 文件的路径。
        tokenizer: The tokenizer instance.
        max_length (int): tokenizer 的最大序列长度。
        num_workers (int): 用于处理的 CPU 核心数，默认为全部可用核心。
    """
    input_path = Path(input_dir)
    output_path = Path(output_file)
    # 如果未指定核心数，则使用所有可用的 CPU 核心
    if num_workers is None:
        num_workers = os.cpu_count()
    print(f"正在 {input_path} 中递归搜索并加载所有 .jsonl 文件...")
    # 使用 glob 模式递归查找所有 .jsonl 文件
    data_files_pattern = str(input_path / '**/*.jsonl')
    # 加载所有文件到一个数据集中
    # cache_dir 可以指定一个位置来存储处理过程中的缓存文件，避免占满默认缓存区
    raw_dataset = load_dataset(
        "json",
        data_files=data_files_pattern,
        cache_dir="./.datasets_cache"
    )
    print(f"文件加载完成。数据集包含 {len(raw_dataset['train'])} 条记录。")
    print(f"开始使用 {num_workers} 个核心进行并行分词...")
    # 使用 map 方法进行并行分词
    # 我们传入一个 functools.partial 来固定 tokenizer 和 max_length 参数
    tokenized_dataset = raw_dataset['train'].map(
        partial(tokenize_batch, tokenizer=tokenizer, max_length=max_length),
        batched=True,
        num_proc=num_workers,
        remove_columns=raw_dataset['train'].column_names,  # 移除原始列
        desc="正在并行分词"
    )
    print("分词处理完成。正在统计 Token 总数...")
    # 从临时列中计算总 Token 数
    total_tokens = sum(tokenized_dataset["token_count"])
    # 移除用于统计的临时列
    final_dataset = tokenized_dataset.remove_columns(["token_count"])
    print(f"Token 总数统计完成。正在将结果保存到 {output_path}...")
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 将处理后的数据集保存为单个 jsonl 文件
    final_dataset.to_json(output_path, force_ascii=False)
    print("所有处理已完成。")
    return total_tokens

# --- 脚本主入口 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理 JSONL 文件，进行分词并输出 tokenized 文件。")
    parser.add_argument("--input_dir", type=str, required=True, help="输入 JSONL 文件夹路径")
    parser.add_argument("--output_file", type=str, required=True, help="输出 tokenized JSONL 文件路径")
    parser.add_argument("--tokenizer_model", type=str, required=True, help="Tokenizer 模型路径 (e.g., 'path/to/your/LLaDA-8B-Base')")
    parser.add_argument("--max_length", type=int, default=8192, help="最大 token 长度 (默认: 8192)")
    parser.add_argument("--num_workers", type=int, default=192, help="CPU 核心数 (默认: 192)")
    
    args = parser.parse_args()
    
    # 加载 tokenizer
    TOKENIZER = AutoTokenizer.from_pretrained(args.tokenizer_model)
    
    total_tokens_processed = process_files_with_datasets(
        args.input_dir,
        args.output_file,
        TOKENIZER,
        max_length=args.max_length,
        num_workers=args.num_workers
    )
    
    # 以十亿（Billion）为单位打印最终结果
    print(f"处理的总 Token 数量: {total_tokens_processed / 1e9:.4f}B")