run baseline:
```
CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.run --nproc_per_node=1 --run-path flux/run_flux_stepparallelism.py
```

run parastep:
```
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.run --nproc_per_node=2 --run-path flux/run_flux_stepparallelism.py
```