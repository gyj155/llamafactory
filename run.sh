#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
/home/yejie/miniconda3/envs/lf/bin/llamafactory-cli train config/qwen3vl_lora_sft.yaml
