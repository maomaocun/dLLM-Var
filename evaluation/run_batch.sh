export LOG_PATH=/path/to/logs/dLLM-Var/evaluation
export BLOCK_LENGTH=64 
export MAX_GEN_LENGTH=1024
export ENABLE_EDITING=False # the model now ((2025.11.3) released does not support editing
export THRESHOLD=0.9
export PRETRAINED_PATH=\path\to\dLLM-Var
export BATCH_SIZE=1
bash /mnt/innovator/code/yangyicun/dLLM-Pro/evaluation/run_eval_block_diffusion_template.sh

