#!/bin/bash
TRAIN_CONFIG=$1
DISTRIBUTED_ARGS="
    --nproc_per_node ${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)} \
    --nnodes ${WORLD_SIZE:-1} \
    --node_rank ${RANK:-0} \
    --master_addr ${MASTER_ADDR:-127.0.0.1} \
    --master_port ${MASTER_PORT:-23456}
    "
    
set -ex
torchrun $DISTRIBUTED_ARGS src/train.py $TRAIN_CONFIG