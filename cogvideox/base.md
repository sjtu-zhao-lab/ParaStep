run baseline
```
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.run --nproc_per_node=1 --run-path cogvideox/run_cogvideox_stepparallelism.py
```

run ParaStep(degree of parallelism is 2)
```
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.run --nproc_per_node=2 --run-path cogvideox/run_cogvideox_stepparallelism.py
```