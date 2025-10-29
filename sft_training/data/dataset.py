import glob
import os
import random
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset
from datasets import load_dataset
from typing import List, Any
from tqdm import tqdm
import threading
import queue

import torch
from transformers import DefaultDataCollator, AutoTokenizer

class dLLMDataCollator(DefaultDataCollator):
    """
    Adds the forward noising process to the batch.
    - mask_forward_process: Applies masking based on a noise schedule 't'.
    - edit_forward_process: Applies token editing. The edit ratio is now
      INVERSELY correlated with the noise level 't'.
    """

    def __init__(self, tokenizer, max_length, mask_token_id=126336):
        super().__init__()
        self.tokenizer = tokenizer
        self.mask_token_id = mask_token_id
        self.vocab_size = tokenizer.vocab_size
        self.max_length = max_length
        self.a = 0.1
        self.b = 0.9

    def mask_forward_process(self, batch, eps=1e-3):
        input_ids = batch["input_ids"]
        B, N = input_ids.shape
        
        if "t" not in batch:
            t = torch.rand((B,), device=input_ids.device)
        else:
            t = batch["t"]

        t = (self.b - self.a) * t + self.a
        t_expanded = t[:, None].repeat(1, N)

        mask_indices = torch.rand((B, N), device=input_ids.device) < t_expanded
        # 126081 eos token id
        eos_indices = (input_ids == 126348) | (input_ids == 126081)
        mask_indices = mask_indices | eos_indices

        noisy_batch = torch.where(mask_indices, self.mask_token_id, input_ids)
        
        return noisy_batch, t, mask_indices,eos_indices

    def __call__(self, batch):
        clean_batch = batch["input_ids"].clone()
        noisy_batch, t_per_example, mask_indices,_ = self.mask_forward_process(batch)
        prompt_mask_index = batch["labels"] == -100
        noisy_batch[prompt_mask_index] = clean_batch[prompt_mask_index]
        batch["input_ids"] = noisy_batch.long()
        batch["labels"][~mask_indices] = -100
        return batch


class StreamingSFTConcatDataset(IterableDataset):
    """
    一个【永不耗尽】并采用【异步预加载】的流式 SFT 数据集。

    核心特性:
    - **异步数据加载 (Asynchronous Pre-fetching)**: 引入了一个后台线程和一个数据队列
      (buffer)，专门负责数据的加载和预处理。主训练线程可以直接从队列中获取已准备好的
      数据块，从而避免了 I/O 等待，显著提升了训练效率和 GPU 利用率。
    - **永不耗尽 (Perpetual)**: 当数据集遍历一次后，会自动从头开始新的循环（epoch），
      确保训练可以达到任意设定的 `max_steps`。
    - **动态随机化 (Dynamic Shuffling)**: 每个新的循环（epoch）都会使用不同的随机种子
      进行数据打乱，这比单次 shuffle 更能增强模型的泛化能力。
    - **分布式安全 (DDP-Safe)**: 内置了对 PyTorch 分布式数据并行 (DDP) 的支持。
      它会自动检测分布式环境，并为每个进程（rank）分配一个唯一、不重叠的数据分片。
    - **流式处理 (Streaming)**: 直接从磁盘流式读取数据，无需将整个数据集加载到内存中，
      非常适合处理超大规模的数据集。
    - **实时拼接与分块 (On-the-fly Concatenation and Chunking)**: 将多条数据实时
      拼接成一个连续的 token 流，然后切割成固定长度 (`max_length`) 的数据块进行输出，
      以提高训练效率。
    - **调试模式 (Debug Mode)**: 支持打印详细的后台数据加载情况，如缓冲区状态。
    """
    def __init__(
        self,
        data_files: str,
        tokenizer: Any,
        max_length: int = 8192,
        batch_size: int = 2,
        shuffle_buffer_size: int = 10000,
        seed: int = 42,
        streaming: bool = True,
        disable_tqdm: bool = True,
        buffer_size: int = 10000,
        debug: bool = False
    ):
        """
        初始化数据集。

        Args:
            data_files (str): 数据文件路径。可以是一个 .jsonl 文件，也可以是一个包含多个 .jsonl 文件的目录。
            tokenizer (Any): Hugging Face Tokenizer 实例。
            max_length (int): 输出 token 块的固定长度。
            shuffle_buffer_size (int): 流式 shuffle 的缓冲区大小。数值越大，随机性越好，但内存占用略高。
            seed (int): 用于 shuffle 和模板选择的基础随机种子。
            streaming (bool): 是否以流式模式加载数据。对于大文件建议始终为 True。
            disable_tqdm (bool): 是否禁用 tqdm 进度条。
            buffer_size (int): 异步加载队列的大小。这个队列用于存储预处理好的数据块。
            debug (bool): 是否开启调试模式。开启后会打印缓冲区状态等信息。
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length 
        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.streaming = streaming
        self.disable_tqdm = disable_tqdm
        self.debug = debug
        self.data_collator = dLLMDataCollator(tokenizer=tokenizer, max_length=max_length)
        # 初始化数据缓冲区和线程控制
        self.buffer = queue.Queue(maxsize=buffer_size)
        self.data_loader_thread = None
        self._stop_event = threading.Event()
        if self.debug:
            self.items_added_to_buffer = 0
            self.items_retrieved_from_buffer = 0

        # 初始化分布式训练环境信息
        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            print(f"DDP environment detected. Rank: {self.rank}, World Size: {self.world_size}")
        else:
            self.rank = 0
            self.world_size = 1
            print("No DDP environment detected. Running in single-process mode.")

        # 在 __init__ 中，我们只解析文件路径模式，不实际加载数据
        if os.path.isdir(data_files):
                print(f"Loading and shuffling .jsonl files from directory: {data_files}")
                
                # Get the list of all files
                all_files = glob.glob(os.path.join(data_files, "*.jsonl"))
                
                # SHUFFLE THE LIST IN-PLACE. THIS FUNCTION RETURNS NONE.
                random.shuffle(all_files)
                
                # Now, assign the shuffled list to the instance variable
                self.data_files_pattern: list = all_files
                
        else:
            print(f"Loading data from single file: {data_files}")
            self.data_files_pattern: str = data_files
        
        # 启动后台数据加载线程
        self.start_loader_thread()

    def _data_loader_worker(self):
        """
        核心的数据加载器工作函数，在后台线程中运行。
        它负责从数据源读取、处理数据，并将结果放入缓冲区队列。
        """
        epoch = 0
        while not self._stop_event.is_set():  # 检查停止信号
            # 为每个 rank 和每个 epoch 设置唯一的随机种子
            current_seed = self.seed + self.rank + epoch
            random.seed(current_seed)
            if self.rank == 0:
                print(f"Starting new data cycle (epoch {epoch}). Base seed for this cycle: {self.seed + epoch}")
            
            stream_dataset = load_dataset(
                "json",
                data_files=self.data_files_pattern,
                split="train",
                streaming=self.streaming,
            )

            # 使用动态种子进行 shuffle
            if self.shuffle_buffer_size > 1:
                sharded_dataset = stream_dataset.shuffle(
                    buffer_size=self.shuffle_buffer_size,
                    seed=self.seed + epoch
                )
            else:
                sharded_dataset = stream_dataset.shuffle(
                    seed=self.seed + epoch
                )

            token_buffer = []
            label_buffer = []
            chunk_ids_list = []
            chunk_label_ids_list = []
            is_tqdm_disabled = (self.rank != 0) or self.disable_tqdm
            num_instruction = 0
            for example in tqdm(sharded_dataset, desc=f"Epoch {epoch} loading", disable=is_tqdm_disabled):
                if self._stop_event.is_set():
                    break # 如果接收到停止信号，则中断数据加载

                input_ids_per_sample = example.get("input_ids", [])
                prompt_length = example.get("prompt_length", 0)

                # --- 修改开始 ---
                # 1. 创建 label 列表，它是 input_ids 的一个副本
                label_ids_per_sample = list(input_ids_per_sample)
                
                # 2. 将 prompt 部分的 label 设置为 -100，以便在计算损失时被忽略
                #    确保 prompt_length 不会超出列表范围
                actual_prompt_length = min(prompt_length, len(label_ids_per_sample))
                label_ids_per_sample[:actual_prompt_length] = [-100] * actual_prompt_length
                # --- 修改结束 ---

                label_buffer.extend(label_ids_per_sample)
                token_buffer.extend(input_ids_per_sample)
                num_instruction += 1

                while len(token_buffer) >= self.max_length:
                    if self._stop_event.is_set():
                        break
                    chunk_ids = token_buffer[:self.max_length]
                    token_buffer = token_buffer[self.max_length:]

                    label_chunk_ids = label_buffer[:self.max_length]
                    label_buffer = label_buffer[self.max_length:]

                    input_ids = torch.tensor(chunk_ids, dtype=torch.long).unsqueeze(0)
                    chunk_ids_list.append(input_ids)

                    label_ids = torch.tensor(label_chunk_ids, dtype=torch.long).unsqueeze(0)
                    chunk_label_ids_list.append(label_ids)

                    if len(chunk_ids_list) == self.batch_size:
                        input_ids = torch.cat(chunk_ids_list)
                        label_ids = torch.cat(chunk_label_ids_list)
                        
                        # --- 修改开始 ---
                        # 将 labels 添加到 data_chunk 字典中
                        data_chunk = {
                            "input_ids": input_ids,
                            "labels": label_ids,
                            "num_instruction": torch.full(input_ids.shape, num_instruction, device=input_ids.device)
                        }
                        # --- 修改结束 ---
                        
                        data_chunk = self.data_collator(data_chunk)
                        # Debug: 检查缓冲区是否已满
                        if self.debug and self.buffer.full():
                            print(f"Rank {self.rank} [DEBUG]: Buffer is full (size: {self.buffer.qsize()}/{self.buffer.maxsize}). Data loader is waiting...")
                        
                        # 将处理好的数据块放入队列
                        self.buffer.put(data_chunk)
                        
                        # 清空列表以准备下一个批次
                        chunk_ids_list = []
                        chunk_label_ids_list = []
                        num_instruction = 0

                    # Debug: 打印放入物品后的状态
                    if self.debug:
                        self.items_added_to_buffer += 1
                        if self.items_added_to_buffer % 200 == 0: # 每200个item打印一次
                            print(f"Rank {self.rank} [DEBUG]: Added item #{self.items_added_to_buffer}. Buffer size: {self.buffer.qsize()}/{self.buffer.maxsize}")

            if self._stop_event.is_set():
                print(f"Rank {self.rank}: Data loader worker received stop signal and is exiting.")
                break
                
            if self.rank == 0:
                print(f"Epoch {epoch + 1} data stream finished.")
            epoch += 1
        
        # 线程结束前，放入一个哨兵值，以通知迭代器停止
        self.buffer.put(None)
    def __iter__(self):
        """
        返回数据生成器。
        它从缓冲区队列中获取数据。
        """
        while True:
            # 从队列中获取数据，如果队列为空则会阻塞
            item = self.buffer.get()
            
            # Debug: 打印获取物品后的状态
            if self.debug and item is not None:
                self.items_retrieved_from_buffer += 1
                if self.items_retrieved_from_buffer % 200 == 0: # 每200个item打印一次
                    print(f"Rank {self.rank} [DEBUG]: Retrieved item #{self.items_retrieved_from_buffer}. Buffer size: {self.buffer.qsize()}/{self.buffer.maxsize}")

            if item is None:  # 遇到哨兵值，表示数据流结束
                break
            yield item

    def start_loader_thread(self):
        """启动后台加载线程"""
        if self.data_loader_thread is None or not self.data_loader_thread.is_alive():
            self._stop_event.clear()
            self.data_loader_thread = threading.Thread(target=self._data_loader_worker, daemon=True)
            self.data_loader_thread.start()
            if self.rank == 0:
                print(f"Started data loader background thread.")

    def stop_loader_thread(self):
        """停止后台加载线程"""
        if self.data_loader_thread and self.data_loader_thread.is_alive():
            self._stop_event.set()
            # 清空队列以释放可能阻塞的 put 操作
            while not self.buffer.empty():
                try:
                    self.buffer.get_nowait()
                except queue.Empty:
                    break
            self.data_loader_thread.join(timeout=5)
            if self.rank == 0:
                print(f"Stopped data loader background thread.")

    def __del__(self):
        """在对象销毁时确保线程被停止"""
        self.stop_loader_thread()

