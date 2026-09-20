r"""Experimental FSDP2 customization APIs.

The nonzero-dimension copy functions alias FSDP's defaults; explicit registration
is unnecessary.

.. warning::
    These APIs are experimental. Callback signatures and supported FSDP
    internals may change without backward compatibility.
"""

from torch.distributed.fsdp._fully_shard._fsdp_api import ReduceScatterInput
from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
    _default_all_gather_output_fn as all_gather_output_fn_for_nonzero_dim_shards,
    _default_reduce_scatter_input_fn as reduce_scatter_input_fn_for_nonzero_dim_shards,
)


__all__ = [
    "ReduceScatterInput",
    "all_gather_output_fn_for_nonzero_dim_shards",
    "reduce_scatter_input_fn_for_nonzero_dim_shards",
]
