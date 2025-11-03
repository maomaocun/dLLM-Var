source /path/to/miniconda3/etc/profile.d/conda.sh # adjust the path accordingly

conda activate /path/to/miniconda3/envs/TE # adjust the path accordingly

export HF_ENDPOINT=https://hf-mirror.com

export HF_HOME="/path/to/.cache/huggingface" # adjust the path accordingly
export HF_DATASETS_OFFLINE=1

cd /path/to/dLLM-Pro/evaluation # adjust the path accordingly


echo ${PRETRAINED_PATH}
echo ${LOG_PATH}
echo ${BLOCK_LENGTH}
echo ${ENABLE_EDITING}
echo ${THRESHOLD}

accelerate launch --config_file accelerate_config.yaml evaluation_script.py -m lm_eval --model LLaDA --tasks gsm8k --batch_size ${BATCH_SIZE} \
--model_args "pretrained=${PRETRAINED_PATH},enable_editing=${ENABLE_EDITING}" \
--gen_kwargs "block_length=${BLOCK_LENGTH},max_gen_length=${MAX_GEN_LENGTH},confidence_threshold=${THRESHOLD}"  \
--num_fewshot 4  \
--output_path /${LOG_PATH}/block_diffusion/gsm8k_log \
--log_samples \



accelerate launch --config_file accelerate_config.yaml  evaluation_script.py -m lm_eval --model LLaDA --tasks gpqa_main_generative_n_shot  --batch_size ${BATCH_SIZE} \
--model_args "pretrained=${PRETRAINED_PATH},enable_editing=${ENABLE_EDITING}" \
--gen_kwargs "block_length=${BLOCK_LENGTH},max_gen_length=${MAX_GEN_LENGTH},confidence_threshold=${THRESHOLD}"  \
--num_fewshot 5  \
--output_path /${LOG_PATH}/block_diffusion/gpqa_log \
--log_samples \
--confirm_run_unsafe_code \
--trust_remote_code \




accelerate launch --config_file accelerate_config.yaml evaluation_script.py -m lm_eval --model LLaDA --tasks mmlu_generative --batch_size ${BATCH_SIZE} \
--model_args "pretrained=${PRETRAINED_PATH},enable_editing=${ENABLE_EDITING}" \
--gen_kwargs "block_length=1,max_gen_length=1,confidence_threshold=${THRESHOLD}" \
--num_fewshot 5  \
--output_path /${LOG_PATH}/block_diffusion/mmlu_log \
--log_samples \




accelerate launch --config_file accelerate_config.yaml evaluation_script.py -m lm_eval --model LLaDA --tasks humaneval_instruct --batch_size ${BATCH_SIZE} \
--model_args "pretrained=${PRETRAINED_PATH},enable_editing=${ENABLE_EDITING}" \
--gen_kwargs "block_length=${BLOCK_LENGTH},max_gen_length=${MAX_GEN_LENGTH},confidence_threshold=${THRESHOLD}"  \
--output_path /${LOG_PATH}/block_diffusion/humaneval_log/ \
--log_samples \
--confirm_run_unsafe_code \



accelerate launch --config_file accelerate_config.yaml evaluation_script.py -m lm_eval --model LLaDA --tasks mbpp --batch_size ${BATCH_SIZE} \
--model_args "pretrained=${PRETRAINED_PATH},enable_editing=${ENABLE_EDITING}" \
--gen_kwargs "block_length=${BLOCK_LENGTH},max_gen_length=${MAX_GEN_LENGTH},confidence_threshold=${THRESHOLD}" \
--num_fewshot 3  \
--output_path /${LOG_PATH}/block_diffusion/mbpp_log \
--log_samples \
--confirm_run_unsafe_code \



accelerate launch --config_file accelerate_config.yaml evaluation_script.py -m lm_eval --model LLaDA --tasks bbh --batch_size ${BATCH_SIZE} \
--model_args "pretrained=${PRETRAINED_PATH},enable_editing=${ENABLE_EDITING}" \
--gen_kwargs "block_length=${BLOCK_LENGTH},max_gen_length=${MAX_GEN_LENGTH},confidence_threshold=${THRESHOLD}"  \
--num_fewshot 3  \
--output_path /${LOG_PATH}/block_diffusion/bbh_log \
--log_samples \
--trust_remote_code \



accelerate launch --config_file accelerate_config.yaml evaluation_script.py -m lm_eval --model LLaDA --tasks minerva_math --batch_size ${BATCH_SIZE} \
--model_args "pretrained=${PRETRAINED_PATH},enable_editing=${ENABLE_EDITING}" \
--gen_kwargs "block_length=${BLOCK_LENGTH},max_gen_length=${MAX_GEN_LENGTH},confidence_threshold=${THRESHOLD}"  \
--num_fewshot 4  \
--output_path /${LOG_PATH}/block_diffusion/minerva_math_log \
--log_samples \




accelerate launch --config_file accelerate_config.yaml evaluation_script.py -m lm_eval --model LLaDA --tasks mmlu_pro --batch_size ${BATCH_SIZE} \
--model_args "pretrained=${PRETRAINED_PATH},enable_editing=${ENABLE_EDITING}" \
--gen_kwargs "block_length=${BLOCK_LENGTH},max_gen_length=${MAX_GEN_LENGTH},confidence_threshold=${THRESHOLD}"  \
--output_path /${LOG_PATH}/block_diffusion/mmlu_pro_log \
--log_samples \
--num_fewshot 0  \
