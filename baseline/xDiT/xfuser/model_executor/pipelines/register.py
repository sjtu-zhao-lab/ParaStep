from typing import Dict, Type, Union
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from xfuser.logger import init_logger
from .base_pipeline import xFuserPipelineBaseWrapper

logger = init_logger(__name__)

class xFuserPipelineWrapperRegister:
    _XFUSER_PIPE_MAPPING: Dict[
        Type[DiffusionPipeline], 
        Type[xFuserPipelineBaseWrapper]
    ] = {}

    @classmethod
    def register(cls, origin_pipe_class: Type[DiffusionPipeline]):
        def decorator(xfuser_pipe_class: Type[xFuserPipelineBaseWrapper]):
            if not issubclass(xfuser_pipe_class, xFuserPipelineBaseWrapper):
                raise ValueError(f"{xfuser_pipe_class} is not a subclass of"
                                 f" xFuserPipelineBaseWrapper")
            cls._XFUSER_PIPE_MAPPING[origin_pipe_class] = \
                xfuser_pipe_class
            return xfuser_pipe_class
        return decorator

    @classmethod
    def get_class(
        cls,
        pipe: Union[DiffusionPipeline, Type[DiffusionPipeline]]
    ) -> Type[xFuserPipelineBaseWrapper]:
        if isinstance(pipe, type):
            candidate = None
            candidate_origin = None
            for (origin_model_class, 
                 xfuser_model_class) in cls._XFUSER_PIPE_MAPPING.items():
                if issubclass(pipe, origin_model_class):
                    if ((candidate is None and candidate_origin is None) or 
                        issubclass(origin_model_class, candidate_origin)):
                        candidate_origin = origin_model_class
                        candidate = xfuser_model_class
            if candidate is None:
                raise ValueError(f"Diffusion Pipeline class {pipe} "
                                 f"is not supported by xFuser")
            else:
                return candidate
        elif isinstance(pipe, DiffusionPipeline):
            candidate = None
            candidate_origin = None
            for (origin_model_class, 
                 xfuser_model_class) in cls._XFUSER_PIPE_MAPPING.items():
                if isinstance(pipe, origin_model_class):
                    if ((candidate is None and candidate_origin is None) or 
                        issubclass(origin_model_class, candidate_origin)):
                        candidate_origin = origin_model_class
                        candidate = xfuser_model_class

            if candidate is None:
                raise ValueError(f"Diffusion Pipeline class {pipe.__class__} "
                                 f"is not supported by xFuser")
            else:
                return candidate
        else:
            raise ValueError(f"Unsupported type {type(pipe)} for pipe")