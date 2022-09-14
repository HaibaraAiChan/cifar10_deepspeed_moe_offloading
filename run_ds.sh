#!/bin/bash

deepspeed cifar10_deepspeed.py \
    --deepspeed --deepspeed_config ds_config.json \
    $@ &>./log/ds-bs-128-model-size-MB.log
