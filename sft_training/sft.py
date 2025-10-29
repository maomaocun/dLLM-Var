
from asyncio import streams
import torch
import argparse
from transformers import TrainingArguments
import os
from data import StreamingSFTConcatDataset,dLLMDataCollator
from trainer import dLLMTrainer
from argsparser import ArgsProcessor
from utils import TransformerModelLoader,LoraBuilder
from datasets import load_dataset
from accelerate import Accelerator

def load_data(args, tokenizer):
   
    train_dataset = StreamingSFTConcatDataset(
        data_files=args.train_data, 
        tokenizer=tokenizer, 
        max_length=args.max_length,
        batch_size = args.local_batch_size
    )
    print("Train and Eval datasets are initialized in streaming mode.")
    print(f"Each item from the dataset will have a fixed length of {args.max_length} tokens.")
    
    return train_dataset, None

def train_model(args, model, tokenizer, train_dataset, eval_dataset):
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, args.job_name),
        # num_train_epochs=args.num_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum_steps,
        # eval_strategy=args.evaluation_strategy,
        # eval_steps=args.eval_steps,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        learning_rate=args.learning_rate,
        # load_best_model_at_end=args.load_best_model_at_end,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        bf16=args.bf16,
        report_to=args.report_to,
        remove_unused_columns=args.remove_unused_columns,
        disable_tqdm=False,
        
    )
    trainer = dLLMTrainer(
        model=model,
        args=training_args,
        data_collator=dLLMDataCollator(tokenizer=tokenizer, mask_token_id=126336, max_length=args.max_length),
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
    )

    # Start training
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configuration parser")
    parser.add_argument("--debug",dest="debug",action="store_true",help="debug mode")
    parser.add_argument("--enable_lora",default=False,help="enable lora")
    parser.add_argument("--train_config_path",type=str,default="./config/sft/default_config.yaml",help="Path to the Train YAML configuration file")
    parser.add_argument("--lora_config_path",type=str,default="./config/lora/default_config.yaml",help="Path to the Lora YAML configuration file")
    parser.add_argument("--deepspeed", type=str, default="./config/deepspeed/ds_config_zero3.json", help="Path to DeepSpeed configuration file")
    parser.add_argument("--accelerator_config", type=str, default="./config/accelerate/fp8_config.yaml", help="Path to accelerate configuration file")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    args = parser.parse_args()
    args_processor = ArgsProcessor(args.train_config_path)
    args = args_processor.add_args_from_yaml(args)
    model_loader = TransformerModelLoader(tokenizer_path=args.model_name,model_path=args.model_name)
    tokenizer, model = model_loader.load_model_tokenizer()
    if args.enable_lora:
        lora_args =  argparse.ArgumentParser(description="Lora Configuration parser").parse_args()
        lora_args_processor = ArgsProcessor(args.lora_config_path)
        lora_args = lora_args_processor.add_args_from_yaml(lora_args)
        lora_bulider = LoraBuilder(lora_args)
        model = lora_bulider.get_Lora(model)
    train_dataset, eval_dataset = load_data(args, tokenizer)
    world_size = int(os.getenv("WORLD_SIZE", 8))
    print(f"Global Batch Size: {args.local_batch_size} (local_batch_size) * {args.grad_accum_steps} (grad_accum_steps) * {int(os.getenv("WORLD_SIZE", 8))} (world_size) = {args.local_batch_size * args.grad_accum_steps * int(os.getenv("WORLD_SIZE", 8))}")
    train_model(args,model,tokenizer,train_dataset,eval_dataset)
    