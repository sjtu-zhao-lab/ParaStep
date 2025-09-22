#!/bin/bash
set -x

export PYTHONPATH=$PWD:$PYTHONPATH

# CogVideoX configuration
SCRIPT="latte_example.py"
MODEL_ID="maxin-cn/Latte-1"
INFERENCE_STEP=50

mkdir -p ./results

# CogVideoX specific task args
# TASK_ARGS="--height 768 --width 1360 --num_frames 17"

# CogVideoX parallel configuration
N_GPUS=1
PARALLEL_ARGS="--ulysses_degree 1 --ring_degree 1"
# CFG_ARGS="--use_cfg_parallel"

# Uncomment and modify these as needed
# PIPEFUSION_ARGS="--num_pipeline_patch 8"
# OUTPUT_ARGS="--output_type latent"
# PARALLLEL_VAE="--use_parallel_vae"
# ENABLE_TILING="--enable_tiling"
# COMPILE_FLAG="--use_torch_compile"

torchrun --nproc_per_node=$N_GPUS ./examples/$SCRIPT \
--model $MODEL_ID \
$PARALLEL_ARGS \
--num_inference_steps $INFERENCE_STEP \
--warmup_steps 5 \
--prompt "A handsome cat is crying" \
--height 512 \
--width 512 \
--seed 20 
# --pipefusion_parallel_degree 2
