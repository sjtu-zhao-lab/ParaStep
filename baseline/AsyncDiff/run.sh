# 4.86s
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 --run-path examples/run_sdxl.py --model_n 2 --stride 1
# baseline, 6.19s
# CUDA_VISIBLE_DEVICES=0 python examples/run_sdxl_baseline.py 

# 
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 --run-path examples/test_sd3.py --model_n 2 --stride 1
