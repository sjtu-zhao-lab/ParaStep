import torch.distributed as dist
import torch
# from torch import module

import os
from torch.distributed.device_mesh import init_device_mesh

from .cogvideox_stepParallelism_call import CogVideoX__call, T5stack_call

from diffusers import CogVideoXPipeline

import types
import time

# monkey hack
CogVideoXPipeline.__call__ = CogVideoX__call




class StepParallelism(object):
    def __init__(self, pipeline, step_size, warm_up = 5, reduce_memory=0):
        """
        when set reduce_memory to 1, means offload and load T5 to reduce GPU memory
        when set reduce_memory to 2, means split the T5 encoder(pipeline parallelism) across the GPUs to reduce GPU memory
        """
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

        self.pipeline = pipeline.to(f"cuda:{self._rank}")
        torch.cuda.set_device(f"cuda:{self._rank}")

        self.pipeline.step_mesh = self.step_mesh
        self.pipeline.step_rank = self.step_rank
        self.pipeline.step_size = step_size
        self.pipeline.warm_up = warm_up
        self.pipeline.inference_step_num = 0
        self.pipeline.comm_flag = False
        self.pipeline.round = 0
        self.pipeline.reduce_memory = reduce_memory
        # fixed, for test

        # reduce memory of t5
        if reduce_memory == 2 and self.step_size>1:
            module = self.pipeline.text_encoder.encoder
            module.step_mesh = self.step_mesh
            module.step_rank = self.step_rank
            module.step_size = step_size

            module.cached_result = None
            module.result_structure = None

            block_num = len(module.block)
            block_num_local = block_num // step_size
            start_i = 0
            for i in range(0, step_size):#
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

        dist.barrier()
    

    def clear_cache(self):
        self.pipeline.inference_step_num = 0
        self.pipeline.comm_flag = False
        self.pipeline.round = 0

