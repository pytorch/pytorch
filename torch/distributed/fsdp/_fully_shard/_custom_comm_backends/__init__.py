"""Optional, opt-in custom communication backends for FSDP2.

These backends are not part of the default FSDP2 all-gather / reduce-scatter
path. They are only used when a user explicitly installs one via
:meth:`FSDPModule.set_custom_all_gather` (or the reduce-scatter equivalent), and
may depend on optional third-party runtimes that are imported lazily.
"""

from ._mori_sdma_allgather import MoriSdmaAllGather


__all__ = ["MoriSdmaAllGather"]
