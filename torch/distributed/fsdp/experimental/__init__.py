"""Experimental FSDP2 APIs.

.. warning::
    These APIs are experimental. Callback signatures and supported FSDP
    internals may change without backward compatibility.
"""

from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
    _all_gather_output_fn_with_dim0_views as all_gather_output_fn_with_dim0_views,
    _prepare_reduce_scatter_inputs_with_dim0_views as reduce_scatter_input_fn_with_dim0_views,
)


__all__ = [
    "all_gather_output_fn_with_dim0_views",
    "reduce_scatter_input_fn_with_dim0_views",
]

for _name in __all__:
    _fn = globals()[_name]
    _fn.__module__ = __name__
    _fn.__name__ = _name
    _fn.__qualname__ = _name
