# mypy: allow-untyped-defs
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.debug.comm_mode import CommDebugMode


def get_sharding_prop_cache_info():
    """
    Get the cache info for the sharding propagation cache, used for debugging purpose only.
    This would return a named tuple showing hits, misses, maxsize and cursize of the sharding
    propagator cache.
    """
    return (
        DTensor._op_dispatcher.sharding_propagator.propagate_op_sharding.cache_info()  # type:ignore[attr-defined]
    )
