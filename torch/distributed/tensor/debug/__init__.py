# mypy: allow-untyped-defs
import torch._C
from torch.distributed.tensor.debug._comm_mode import CommDebugMode
from torch.distributed.tensor.debug._visualize_sharding import visualize_sharding


__all__ = ["CommDebugMode", "visualize_sharding"]


def _get_python_sharding_prop_cache_info():
    """
    Get the cache info for the Python sharding propagation cache, used for debugging purpose only.
    This would return a named tuple showing hits, misses, maxsize and cursize of the sharding
    propagator cache. Note that directly calling into the sharding propagator does not share cache
    state with the DTensor dispatch fast path!
    """
    from torch.distributed.tensor._api import DTensor

    return (
        DTensor._op_dispatcher.sharding_propagator.propagate_op_sharding.cache_info()  # type:ignore[attr-defined]
    )


def _get_fast_path_sharding_prop_cache_stats():
    """
    Get a tuple (hits, misses) for the fast path sharding propagation cache, used for debugging
    only.
    """
    return torch._C._get_DTensor_sharding_propagator_cache_stats()


def _clear_python_sharding_prop_cache():
    """
    Clears the cache for the Python sharding propagation cache, used for debugging purpose only.
    """
    from torch.distributed.tensor._api import DTensor

    return (
        DTensor._op_dispatcher.sharding_propagator.propagate_op_sharding.cache_clear()  # type:ignore[attr-defined]
    )


def _clear_fast_path_sharding_prop_cache():
    """
    Clears the cache for the fast path sharding propagation cache, used for debugging purpose only.
    """
    torch._C._clear_DTensor_sharding_propagator_cache()


# Set namespace for exposed private names
CommDebugMode.__module__ = "torch.distributed.tensor.debug"
visualize_sharding.__module__ = "torch.distributed.tensor.debug"
