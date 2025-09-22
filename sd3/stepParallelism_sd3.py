import torch.distributed as dist
import torch
import os
from torch.distributed.device_mesh import init_device_mesh

from .sd3_stepParallelism_call import step_parallelism_call, T5stack_call

from diffusers import StableDiffusion3Pipeline

StableDiffusion3Pipeline.__call__ = step_parallelism_call

import types
import time


class StepParallelism(object):
    def __init__(self, pipeline, step_size, warmp_up=5, reduce_memory_level=0):
        self._rank = int(os.environ["RANK"])
        self._world_size = int(os.environ["WORLD_SIZE"])

        self.step_mesh = init_device_mesh("cuda", (self._world_size, ))
        self.step_rank = self.step_mesh.get_local_rank()

        self.pipeline = pipeline.to(f"cuda:{self._rank}")
        torch.cuda.set_device(f"cuda:{self._rank}")

        self.pipeline._rank = self._rank
        self.pipeline.step_mesh = self.step_mesh
        self.pipeline.step_rank = self.step_rank
        self.pipeline.step_size = step_size
        self.pipeline.warm_up = warmp_up
        self.pipeline.inference_step_num = 0
        self.pipeline.comm_flag = False
        self.pipeline.round = 0

        # reduce gpu memory
        self.pipeline.reduce_memory_level = reduce_memory_level
          
        if self.pipeline.reduce_memory_level == 2:
            module = self.pipeline.text_encoder_3.encoder
            module.step_mesh = self.step_mesh
            module.step_rank = self.step_rank
            module.step_size = step_size

            module.cached_result = None
            module.result_structure = None

            block_num = len(module.block)
            block_num_local = block_num // step_size
            start_i = 0
            for i in range(0, step_size):#分成step_size份
                end_i = start_i + block_num_local
                if (i == step_size - 1) and (end_i != block_num):
                    end_i = block_num 
                if self.step_rank == i:#占有这一部分的block
                    module.local_block = list(range(start_i, end_i))
                else:
                    for to_cpu_index in range(start_i, end_i):
                        if to_cpu_index != 0:# i为0时，不offload
                            module.block[to_cpu_index] = module.block[to_cpu_index].to('cpu')

                start_i = start_i + block_num_local
            torch.cuda.empty_cache()
            print(f"offload layer_module of T5TextEncoder which is not belong to rank{self._rank} to cpu")

            module.forward = types.MethodType(T5stack_call, module)
            

    def clear_cache(self):
        self.pipeline.inference_step_num = 0
        self.pipeline.comm_flag = False
        self.pipeline.round = 0
        torch.cuda.empty_cache()
