export TENSORBOARD_LOGDIR=/path/to/tensorboard/logs

export WORLD_SIZE=${WORLD_SIZE:-1}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-29500}
export GPUPERNODE=${GPUPERNODE:-8}
export RANK=${RANK:-0}
NUM_PROCESSES=$((WORLD_SIZE * GPUPERNODE))
export NUM_PROCESSES=${NUM_PROCESSES:-8}

echo "Distributed Training Configuration:"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "GPUPERNODE: $GPUPERNODE"
echo "RANK: $RANK"
echo "NUM_PROCESSES: $NUM_PROCESSES"

# FP8 training
accelerate launch \
  --config_file ./config/accelerate/fp8_config_zero2.yaml \
  --num_machines $WORLD_SIZE \
  --num_processes $NUM_PROCESSES\
  --main_process_ip $MASTER_ADDR  \
  --main_process_port $MASTER_PORT \
  --machine_rank $RANK \
  ./sft.py

# BF16 training
accelerate launch \
  --config_file ./config/accelerate/config_zero2.yaml \
  --num_machines $WORLD_SIZE \
  --num_processes $NUM_PROCESSES\
  --main_process_ip $MASTER_ADDR  \
  --main_process_port $MASTER_PORT \
  --machine_rank $RANK \
  ./sft.py
 
