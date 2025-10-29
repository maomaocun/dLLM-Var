import torch
import torch.nn.functional as F
from transformers import Trainer
from transformers import DefaultDataCollator
import random
from tqdm import tqdm
import pickle
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
import random
import math  # 新增
import numpy as np # 新增
class dLLMTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_num_instruction = 0 
        self.total_tokens = 0
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """
        Absorbing state diffusion loss computation
        """
        labels,num_instruction = inputs.pop("labels"), inputs.pop("num_instruction")
        self.total_tokens += labels.shape[0]*labels.shape[1]
        self.total_num_instruction+=num_instruction[0][0].item()
        if (self.state.global_step + 1) % (self.args.logging_steps*self.args.gradient_accumulation_steps)== 0:
            self.log({
                "num_instruction": self.total_num_instruction * dist.get_world_size(),
                "total_tokens": self.total_tokens * dist.get_world_size()/1e9,
                "num_gpus":dist.get_world_size(),
            })
        outputs = model(**inputs)
        logits = outputs.logits
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1), reduction="mean")
        return loss if not return_outputs else (loss, outputs)