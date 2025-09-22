run baseline:
```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 --run-path audioLDM2/run_audioldm2_stepparallelism.py
```

run ParaStep(degree of parallelism is 2):
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 --run-path audioLDM2/run_audioldm2_stepparallelism.py
```

run BatchStep(stride size is 2):
```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 --run-path audioLDM2/run_audioldm2_stepparallelism.py --islocal True --predict_len 2
```