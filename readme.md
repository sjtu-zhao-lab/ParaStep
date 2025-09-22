This repo is the official implementation of "Communication-Efficient Diffusion Denoising Parallelization via Reuse-then-Predict Mechanism"(NIPS'25)

ParaStep proposes and utilizes the `Reuse-then-Predict` mechanism to perform parallel denoising to reduce the latency of Diffusion models. For more details, please read our paper.

## Install
```
conda create -n ParaStep python=3.10
conda activate ParaStep
# git clone this repo
cd ParaStep
pip install -r requirements.txt
```
+ The most important thing is: torch==2.5.1, transformers==4.47.1, diffusers==0.32.0
+ If you meet some conflicts or miss some package, just follow the hints in your terminal. And confirm the version of torch, transformers and diffusers is 'correct'


## run-ParaStep
follow the `base.md` in every folders (audioLDM2, CogVideoX and so on).
Notably, the script will autoly download the weights for diffusion models, to control the download path, you can set the HF_HOME in your `~/.bashrc` or `~/.zshrc`
```
export HF_HOME=<Your HF_HOME>
```

For example, for CogVideox:
run baseline
```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 --run-path cogvideox/run_cogvideox_stepparallelism.py
```

run ParaStep(degree of parallelism is 2)
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 --run-path cogvideox/run_cogvideox_stepparallelism.py
```


## run-BatchStep
`BatchStep` transformers the parallel execution across devices in `ParaStep` into batched execution in a single batch.
We implement BatchStep for audioLDM2 which shows a good batching-effect.

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


## evaluation for SD3
evaluate the latency and generation quality of Asyncdiff, xDiT-Pipe(PipeFusion) and ParaStep on SD3.
```
bash experiments/latency_and_performance/sd3/cocoeval/run_all.sh
```

## Acknowledgements
We use the listed repo to perform comparison, thanks for their works!
+ AsyncDiff: https://github.com/czg1225/AsyncDiff
+ xDiT: https://github.com/xdit-project/xDiT.git

We use and modify the listed repo to eval the generation, thanks for their works!
+ https://github.com/gudgud96/frechet-audio-distance/tree/main
+ https://github.com/JunyaoHu/common_metrics_on_video_quality
+ https://github.com/Vchitect/VBench
+ https://github.com/haoheliu/audioldm_eval

## Citation
If you find ParaStep useful in your work, please consider citing our work.
```
@article{wang2025communication,
  title={Communication-Efficient Diffusion Denoising Parallelization via Reuse-then-Predict Mechanism},
  author={Wang, Kunyun and Li, Bohan and Yu, Kai and Guo, Minyi and Zhao, Jieru},
  journal={arXiv preprint arXiv:2505.14741},
  year={2025}
}
```