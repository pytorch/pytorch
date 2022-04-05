from typing import List
import abc


import torch

import torch.distributed._shard.sharding_spec as shard_spec

from .shard import Shard
from .utils import _flatten_tensor_size


class ShardedTensorInterface(torch.Tensor):
    @staticmethod
    def __new__(cls,
                sharding_spec: shard_spec.ShardingSpec,
                *size,
                **kwargs):
        # Use __new__ for logging purposes.
        torch._C._log_api_usage_once("torch.distributed._shard.sharded_tensor")
        sizes = _flatten_tensor_size(size)
        dtype = kwargs['dtype']
        layout = kwargs['layout']
        pin_memory = kwargs['pin_memory']
        requires_grad = kwargs['requires_grad']
        r = torch.Tensor._make_wrapper_subclass(
            cls,
            sizes,
            dtype=dtype,
            layout=layout,
            pin_memory=pin_memory,
            requires_grad=requires_grad
        )
        return r

    # We define this function for two reasons:
    #  - So that this subclass is recognised as a python subclass by the backend
    #  - So that the user gets friendly errors if they use only torch_function but
    #    their subclass is used on the c++ side (by autograd for example)
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        raise RuntimeError(f"A {cls.__name__} object is being used from c++ while calling {func.__module__}.{func.__name__} "
                           "but the there is no custom __torch_dispatch__ implementation for it.")

    @abc.abstractmethod
    def sharding_spec(self) -> shard_spec.ShardingSpec:
        raise NotImplementedError()

    @abc.abstractmethod
    def local_shards(self) -> List[Shard]:
        raise NotImplementedError()

    @abc.abstractmethod
    def reshard(self, resharding_spec: shard_spec.ShardingSpec) -> "ShardedTensorInterface":
        raise NotImplementedError()

    def __hash__(self):
        return id(self)
