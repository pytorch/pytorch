import torch
import torch.distributed as dist

from torch.overrides import get_default_nowrap_functions
from torch.distributed._shard.sharded_tensor.api import ShardedTensor
from torch.distributed import distributed_c10d


class ReplicatedTensor(torch.Tensor):
    """
    ReplicatedTensor represents a tensor which is replicated across the `world_size` and
    has the same value on each rank.

    ReplicatedTensor is a :class:`~torch.Tensor` subclass, and it could be used together
    with ShardedTensor/Tensor together to express different types of computation. The
    inter-op rules defined as (using torch.add as an example op):
        ReplicatedTensor + ReplicatedTensor = ReplicatedTensor
        ReplicatedTensor + torch.Tensor = torch.Tensor
        ReplicatedTensor + ShardedTensor = ShardedTensor

    NOTE: We do not gurantee equal content of ReplicatedTensor across nodes after its
    construction. Although we defined proper inter-op rules to make sure ReplicatedTensor
    stays the same, there's no enforcement on it (i.e. if you manually modify content on
    some ranks, the modified value will not automatically get synced to other nodes). If
    you wish to manually validate tensors are the same across ranks, use `validate()`.

    """
    def __new__(cls, data=None):
        if data is None:
            data = torch.empty(0)
        return torch.Tensor._make_subclass(cls, data)

    def __repr__(self):
        return f"ReplicatedTensor({super(ReplicatedTensor, self).__repr__()})"

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        new_args = []
        new_kwargs = {}

        # check if args/kwargs have tensor (non-replicated) operands, we have to do
        # this so that we can detect there's a torch.Tensor operand and thus not
        # converting results back to ReplicatedTensor
        has_tensor = False
        for arg in args:
            if isinstance(arg, torch.Tensor) and not isinstance(arg, ReplicatedTensor):
                has_tensor = True

        if kwargs is not None:
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor) and not isinstance(v, ReplicatedTensor):
                    has_tensor = True

        # do dispatch base on the inter-op rules we defined
        for arg in args:
            if isinstance(arg, ShardedTensor):
                # redispatch to ShardedTensor
                # TODO: handle ShardedTensor inter-op with ReplicatedTensor
                return arg.__torch_function__(func, types, args, kwargs)
            else:
                new_args.append(arg)


        if kwargs is not None:
            for k, v in kwargs.items():
                if isinstance(v, ShardedTensor):
                    # redispatch to ShardedTensor
                    # TODO: handle ShardedTensor inter-op with ReplicatedTensor
                    return v.__torch_function__(func, types, args, kwargs)
                else:
                    new_kwargs[k] = v

        # We cann't do super().__torch_function__() as it implicitly convert the result
        # back to tensor subclasses, where in our case, we need to control the output type
        # base on the inter-op rules we defined.
        with torch._C.DisableTorchFunction():
            rs = func(*new_args, **new_kwargs)
            if func in get_default_nowrap_functions():
                return rs
            if not has_tensor and isinstance(rs, torch.Tensor) and not isinstance(rs, cls):
                # if it does not have tensor operands and does not get dispatched to ShardedTensor
                # __torch_function__, we return a ReplicatedTensor according to our inter-op rule
                rs = rs.as_subclass(cls)

            return rs

    def validate(self, process_group=None) -> bool:
        """
        Validate the ReplicatedTensor is legit by all gathering tensors on all ranks
        and check to make sure they are the same.

        If there's some ranks with different values, a ValueError will be raised.

        Keyword args:
            process_group (ProcessGroup, optional): The process group to work on. If None,
                the default process group will be used.

        Returns:
            True if validation succeed.
        """
        process_group = (
            process_group
            if process_group is not None
            else distributed_c10d._get_default_group()
        )
        world_size = dist.get_world_size(process_group)
        current_rank = dist.get_rank(process_group)

        tensors_on_rank = [torch.empty_like(self) for _ in range(world_size)]

        dist.all_gather(tensors_on_rank, self, group=process_group)
        # validate and check if all tensors are equal
        for rank, tensor in enumerate(tensors_on_rank):
            if not torch.allclose(self, tensor):
                raise ValueError(
                    f"ReplicatedTensor have different values on rank {current_rank} and {rank}")

        return True
