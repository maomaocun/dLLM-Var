# Config Directory Overview

This directory contains configuration files for different training and fine-tuning strategies in the dLLM-Factory project. Each subfolder is dedicated to a specific training method or framework, and contains YAML files that define hyperparameters, model settings, and distributed training options.

## Structure

- `sft/`  
  Contains configuration files for Supervised Fine-Tuning (SFT) tasks.
  - `default_config.yaml`: Comprehensive configuration for SFT training with the following parameters:
    - `model_name`: Name of the pretrained model to use (e.g., "GSAI-ML/LLaDA-8B-Instruct").
    - `local_batch_size`: Batch size per device for training.
    - `max_length`: Maximum sequence length for tokenization.
    - `num_epochs`: Number of training epochs.
    - `learning_rate`: Optimizer learning rate.
    - `grad_accum_steps`: Gradient accumulation steps for effective batch size scaling.
    - `output_dir`: Directory to save model checkpoints and logs.
    - `job_name`: Task name for identification and logging.
    - `train_data`: Path to the training dataset.
    - `max_grad_norm`: Gradient clipping threshold for training stability.
    - `weight_decay`: Weight decay parameter for regularization.
    - `evaluation_strategy`: Strategy for model evaluation ("steps", "epoch", etc.).
    - `eval_steps`: Number of training steps between evaluations.
    - `logging_steps`: Number of training steps between logging outputs.
    - `save_steps`: Number of training steps between model checkpoints.
    - `save_total_limit`: Maximum number of checkpoints to keep.
    - `load_best_model_at_end`: Whether to load the best model at the end of training.
    - `bf16`: Whether to use bfloat16 precision for training.
    - `report_to`: Reporting tool configuration (e.g., "none", "wandb").
    - `remove_unused_columns`: Whether to remove unused columns from the dataset.

- `accelerate/`  
  Contains configuration files for distributed training using the HuggingFace Accelerate and DeepSpeed frameworks.
  - `lora_config.yaml`: Configuration for LoRA (Low-Rank Adaptation) training with DeepSpeed. Includes settings for compute environment, DeepSpeed offloading, ZeRO optimization, distributed type, mixed precision, number of processes, and more.
  - `full_param_config.yaml`: A more comprehensive DeepSpeed configuration, including optimizer parameters, bf16 mixed precision, ZeRO optimization details, logging frequency, and distributed training options.

- `lora/`  
  Contains LoRA-specific configuration files for parameter-efficient fine-tuning.
  - `default_config.yaml`: Defines LoRA rank, alpha, target modules for injection (e.g., `q_proj`, `k_proj`, `v_proj`), dropout rate, bias training, and task type (e.g., `CAUSAL_LM`).

## Usage

- Modify the YAML files in each subfolder to adjust training hyperparameters and distributed training settings according to your hardware and task requirements.
- These configuration files are referenced by training scripts in the main project to initialize models, optimizers, and distributed environments.
- For SFT training, start with `sft/default_config.yaml` and adjust parameters based on your model size, available GPU memory, and training objectives.

## Configuration Tips

### SFT Training Parameters
- **Batch Size**: Start with `local_batch_size: 1` for large models and increase if memory allows.
- **Learning Rate**: Use `1e-5` as a starting point for fine-tuning, adjust based on model size and task.
- **Gradient Accumulation**: Use `grad_accum_steps` to effectively increase batch size without increasing memory usage.
- **Precision**: Enable `bf16: true` for faster training and reduced memory usage on supported hardware.
- **Evaluation**: Set `evaluation_strategy: "steps"` and `eval_steps: 100` for regular model evaluation.
- **Checkpointing**: Configure `save_steps` and `save_total_limit` based on your storage capacity and training duration.

## Notes
- Comments in the YAML files provide further explanations for each field.
- For more details on each training strategy, refer to the main project documentation.
- Monitor GPU memory usage and adjust batch sizes and gradient accumulation steps accordingly.
- Consider using mixed precision training (`bf16: true`) for better performance on modern GPUs.
