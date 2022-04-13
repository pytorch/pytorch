from typing import List
import abc


import torch

import torch.distributed._shard.sharding_spec as shard_spec

from .shard import Shard
from .utils import _flatten_tensor_size


class ShardedTensorInterface(torch.Tensor):
    """
    ShardedTensorInterface is an interface of the ShardedTensor abstraction
    which represents Tensors that are sharded across multiple devices and
    multiple processes.

    :class:`ShardedTensor` is a concrete implementation of this interface
    and attached local shards with real data/memory.
    """

    @staticmethod
    def __new__(cls,
                sharding_spec: shard_spec.ShardingSpec,
                *size,
                **kwargs):
        # Use __new__ to construct a wrapper tensor, for recording tensor
        # properties and logging purposes.
        torch._C._log_api_usage_once("torch.distributed._shard.sharded_tensor")
        sizes = _flatten_tensor_size(size)
        dtype = kwargs['dtype']
        layout = kwargs['layout']
        pin_memory = kwargs['pin_memory']
        requires_grad = kwargs['requires_grad']
        r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
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
        """
        Returns the ShardingSpec for the subclass instance of ShardedTensorInterface.

        Subclass of :class:`ShardedTensorInterface` could re-use existing :class:`ShardingSpec`
        implementations like :class:`ChunkShardingSpec`, or implement its own custom spec.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def local_shards(self) -> List[Shard]:
        """
        Returns a list of :class:`Shard' corresponding to the local shards for this rank.
        Returns an empty list if the current rank does not host any shards.

        Subclass of :class:`ShardedTensorInterface` should implement this method if it
        choose to re-use :class:`Shard` for saving local shards.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def reshard(self, resharding_spec: shard_spec.ShardingSpec) -> "ShardedTensorInterface":
        """
        Reshard a sharded tensor given the ``resharding_spec``.

        Subclass of :class:`ShardedTensorInterface` should implement this method if it choose
        to provide the resharding functionality.
        """
        raise NotImplementedError()

    def __hash__(self):
        """
        By default implementation to define behavior when `hash()` is called on an instance
        of class.

        Subclass of :class:`ShardedTensorInterface` could override this method if it choose
        to return a different result.
        """
        return id(self)
