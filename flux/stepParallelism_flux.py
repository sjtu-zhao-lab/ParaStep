import torch.distributed as dist
import torch

import os
from torch.distributed.device_mesh import init_device_mesh

from flux.flux_stepParallelism_call import step_parallelism_call

from diffusers import FluxPipeline

# monkey hack
FluxPipeline.__call__ = step_parallelism_call




class StepParallelism(object):
    def __init__(self, pipeline, step_size, warm_up = 5, step_num = 50):
        self._rank = int(os.environ["RANK"])
        self._world_size = int(os.environ["WORLD_SIZE"])

        if self._world_size != step_size:
            print("_world_size must be same with step_size")
            exit()
        
        # StepParallelism
        self.step_mesh = init_device_mesh("cuda", (self._world_size, ))
        self.step_rank = self.step_mesh.get_local_rank()
        self.step_size = step_size
        self.warm_up = warm_up
        self.step_num = step_num

        self.pipeline = pipeline.to(f"cuda:{self._rank}")
        torch.cuda.set_device(f"cuda:{self._rank}")

        self.pipeline.step_mesh = self.step_mesh
        self.pipeline.step_rank = self.step_rank
        self.pipeline.step_size = step_size
        self.pipeline.warm_up = warm_up
        self.pipeline.step_num = step_num
        self.pipeline.inference_step_num = 0
        self.pipeline.round = 0

        dist.barrier()

    def clear_cache(self):
        self.pipeline.inference_step_num = 0
        self.pipeline.round = 0
        torch.cuda.empty_cache()
