import torch
from torch.utils._pytree import tree_map
from typing import Optional

def _basic_validation(op, args=(), kwargs=None):
    """
    Common validation across all ops go in here.
    """
    from torch.distributed._shard.partial_tensor import _PartialTensor
    from torch.distributed._shard.replicated_tensor import ReplicatedTensor
    from torch.distributed._shard.sharded_tensor import ShardedTensor

    if len(args) == 0 and (kwargs is None or len(kwargs) == 0):
        raise ValueError(f" No input for '{op.__name__}'!")

    # Validate types
    has_distributed_tensor = False

    def is_distributed_tensor(e):
        nonlocal has_distributed_tensor
        if isinstance(e, ReplicatedTensor) or isinstance(e, _PartialTensor) or isinstance(e, ShardedTensor):
            has_distributed_tensor = True

    tree_map(is_distributed_tensor, args)
    tree_map(is_distributed_tensor, kwargs)

    if not has_distributed_tensor:
        raise TypeError(
            f"torch function '{op.__name__}', with args: {args} and "
            f"kwargs: {kwargs} are called without any distributed tensor!"
        )

    # Validate all distributed tensors use the same PG.
    cur_pg: Optional[torch.distributed.ProcessGroup] = None

    def validate_pg(e):
        nonlocal cur_pg
        if isinstance(e, ReplicatedTensor) or isinstance(e, _PartialTensor) or isinstance(e, ShardedTensor):
            if cur_pg is not None and e._process_group is not cur_pg:
                raise RuntimeError(
                    'All distributed tensors should use the '
                    'same ProcessGroup if used together in an op.'
                )
            cur_pg = e._process_group

    tree_map(validate_pg, args)
    tree_map(validate_pg, kwargs)

def _register_default_op(op, decorator):
    @decorator(op)
    def tensor_default_op(types, args=(), kwargs=None, pg=None):
        """
        Handles ``__torch_function__`` dispatch for the default tensor ops that
        behave the same as ``torch.Tensor`` such as ``torch.Tensor.shape`` or
        ``torch.Tensor.dtype``. We simply lower to the real op call with
        DisableTorchFunction context like ``torch.Tensor.__torch_function__``
        to avoid recursions.
        """
        if kwargs is None:
            kwargs = {}

        with torch._C.DisableTorchFunction():
            return op(*args, **kwargs)
