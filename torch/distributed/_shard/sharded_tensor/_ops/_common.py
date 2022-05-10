import functools
from torch.distributed._shard.sharded_tensor import (
    sharded_op_impl,
    Shard,
    ShardedTensor,
)
from torch.distributed._shard.partial_tensor import _PartialTensor
from torch.distributed._shard.replicated_tensor import ReplicatedTensor
from torch.utils._pytree import tree_map

def _basic_validation(op, args=(), kwargs=None):
    """
    Common validation across all ops go in here.
    """
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

    # Validate all DistributedTensors use the same PG.
    cur_pg = None

    def validate_pg(e):
        nonlocal cur_pg
        if isinstance(e, ReplicatedTensor) or isinstance(e, _PartialTensor):
            if cur_pg is not None and e.process_group is not cur_pg:
                raise RuntimeError(
                    'All distributed tensors should use the '
                    'same ProcessGroup if used together in an op.'
                )
            cur_pg = e.process_group
        elif isinstance(e, ShardedTensor):
            if cur_pg is not None and e._process_group is not cur_pg:
                raise RuntimeError(
                    'All distributed tensors should use the '
                    'same ProcessGroup if used together in an op.'
                )
            cur_pg = e._process_group

    tree_map(validate_pg, args)
    tree_map(validate_pg, kwargs)

def _sharded_op_common(op, early_stop_func, extra_check):
    """
    Inject sharded tensor op registration with common logics executed before
    different behaviors are done on either local shards or a local tensor.

    Example::
        >>> op = torch.transpose
        >>> @sharded_op_impl(op)
        >>> @_sharded_op_common(op, early_stop_func, extra_check)
        >>> def sharded_tensor_op(types, args, kwargs, process_group):
        >>>   ....
        >>>
        >>> st = sharded_tensor.rand(32, 16)
        >>> st.transpose(1, 2)
        >>> # This will call '_sharded_op_common'

    Args:
        op: The op to be registered and applied to all shards of the st.
        early_stop_func (Callable, optional): the func for early stop.
            Default: if ``None``, no early stop.
        extra_check (Callable, optional): the func for extra condition check.
            Default: if ``None``, no extra check.

    Return:
        func (Callable): Torch function for which we want to provide a sharded
            implementation (ex: torch.transpose)
    """
    def decorator_sharded_func(wrapped_func):
        @functools.wraps(wrapped_func)
        def wrapper(types, args=(), kwargs=None, pg=None):
            _basic_validation(op, args, kwargs)

            st = args[0]
            if kwargs is None:
                kwargs = {}
            if extra_check:
                extra_check(*args, **kwargs)
            if early_stop_func:
                early_stop = early_stop_func(*args, **kwargs)
                if early_stop:
                    return st
            return wrapped_func(types, args, kwargs, pg)

        return wrapper

    return decorator_sharded_func

def _register_sharded_op_on_local_shards(
    op, early_stop_func=None, extra_check=None, customized_func=None
):
    """
    Handles ``__torch_function__`` dispatch for ops which are performed on
    each shard of the sharded tensor such as elementwise op like
    ``torch.nn.functional.gelu`` or ``torch.nn.functional.relu``.

    For more complicated ops, a customized func can be used to generate
    the new shards and sharded tensor size.

    This function expects that the original ShardingSpec for the ShardedTensor
    is preserved irrespective of whether or not a customized function is used.

    Args:
        op: The op to be registered and applied to all shards of the st.
        early_stop_func (Callable, optional): the func for early stop.
            Default: if ``None``, no early stop.
        extra_check (Callable, optional): the func for extra condition check.
            Default: if ``None``, no extra check.
        customized_func (Callable, optional): the func for customized logic
            to generate new shards and sharded tensor size.
            Default: if ``None``, we simply lower to the real op call with
                all local shards of the st.

    Return:
        func (Callable): registered implementation for sharded op for
        ``__torch_function__`` dispatch.
    """
    @sharded_op_impl(op)
    @_sharded_op_common(op, early_stop_func, extra_check)
    def sharded_tensor_op_on_local_shards(types, args=(), kwargs=None, pg=None):
        st = args[0]
        st_metadata = st.metadata()
        local_shards = st.local_shards()
        local_shards_new = []
        if customized_func:
            local_shards_new, st_metadata = customized_func(args, kwargs, pg)
        else:
            for local_shard in local_shards:
                args = (local_shard.tensor, *args[1:])
                local_shards_new.append(
                    Shard(op(*args, **kwargs), local_shard.metadata)
                )
        return ShardedTensor._init_from_local_shards_and_global_metadata(
            local_shards_new,
            st_metadata,
            process_group=pg,
            init_rrefs=st._init_rrefs,
            sharding_spec=st.sharding_spec()
        )
