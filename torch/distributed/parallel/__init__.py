"""Parallel execution utilities for torch.distributed.

Provides multi-threaded parallel map that serializes collective operations
(all_reduce, broadcast, etc.) on the main thread while running Python-side
computation concurrently in worker threads. Requires nogil Python (3.14t+)
for true parallelism.
"""

from ._parallel_mapper import (
    parallel_map,
    parallel_multi_apply,
    parallel_starmap,
    sync_wrap,
)

__all__ = [
    "parallel_map",
    "parallel_starmap",
    "parallel_multi_apply",
    "sync_wrap",
]
