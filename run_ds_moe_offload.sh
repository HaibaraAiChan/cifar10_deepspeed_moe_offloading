#!/bin/bash -xv

# Number of nodes
NUM_NODES=1
# Number of GPUs per node
NUM_GPUS=4
# Size of expert parallel world (should be less than total world size)
EP_SIZE=4
# Number of total experts
EXPERTS=12

deepspeed --num_nodes=${NUM_NODES} --num_gpus=${NUM_GPUS} cifar10_deepspeed.py \
	--log-interval 10 \
	--deepspeed \
	--deepspeed_config ds_config_offload.json \
	--moe \
	--ep-world-size ${EP_SIZE} \
	--num-experts ${EXPERTS} \
	--top-k 1 \
	--noisy-gate-policy 'RSample' \
	--moe-param-group \
	&> ./log/cifar-moe-offload-epoch-1-nb-64-ep-4412-MB.log
