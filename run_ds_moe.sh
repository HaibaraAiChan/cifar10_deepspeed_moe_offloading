#!/bin/bash

# Number of nodes
NUM_NODES=1
# Number of GPUs per node
NUM_GPUS=4
# Size of expert parallel world (should be less than total world size)
EP_SIZE=4
# Number of total experts
EXPERTS=1024

deepspeed --num_nodes=${NUM_NODES} --num_gpus=${NUM_GPUS} cifar10_deepspeed.py \
	--log-interval 2 \
	--deepspeed \
	--deepspeed_config ds_config.json \
	--moe \
	--ep-world-size ${EP_SIZE} \
	--num-experts ${EXPERTS} \
	--top-k 1 \
	--noisy-gate-policy 'RSample' \
	--moe-param-group   \
	&> ./log/cifar-ds-moe-ep-${NUM_GPUS}${EP_SIZE}${EXPERTS}-MB-model-size-bs-16-mem.log
