# mypy: ignore-errors
"""
Side tables for storing non-graphable arguments that need to be passed through FX graphs.

This module provides a mechanism similar to the Triton kernel side table pattern
(see torch/_higher_order_ops/triton_kernel_wrap.py) for storing arguments that cannot
be directly embedded in FX graphs (e.g., DTensor placements, device meshes, etc.).

Instead of smuggling these values into the graph via closures, we:
1. Store them in a global side table with an integer index
2. Pass the index through the graph
3. Look up the actual values at runtime using the index

This approach enables:
- Proper serialization/caching of FX graphs
- Reproducible graph construction
- Clean separation of graphable vs non-graphable data
"""

import threading
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import torch.fx


class ArgsSideTable:
    """
    A thread-safe side table for storing function arguments that cannot be
    directly embedded in FX graphs.

    This is used for operations like DTensor's redistribute() and to_local()
    which take non-primitive arguments (e.g., Placement objects) that cannot
    be represented as FX graph nodes.
    """

    def __init__(self) -> None:
        self._args_to_id: dict[tuple[tuple[Any, ...], tuple[tuple[str, Any], ...]], int] = {}
        self._id_to_args: dict[int, tuple[tuple[Any, ...], dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def add_args(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> int:
        """
        Add args and kwargs to the side table and return an index.

        If the same args/kwargs have been added before, returns the existing index.
        """

        def make_hashable(value: Any) -> Any:
            """Convert unhashable types to hashable equivalents for key generation."""
            if isinstance(value, list):
                return tuple(make_hashable(v) for v in value)
            if isinstance(value, dict):
                return tuple(sorted((k, make_hashable(v)) for k, v in value.items()))
            return value

        # Create a hashable key from args and kwargs
        hashable_args = tuple(make_hashable(a) for a in args)
        kwargs_tuple = tuple(sorted((k, make_hashable(v)) for k, v in kwargs.items()))
        key = (hashable_args, kwargs_tuple)

        with self._lock:
            if key in self._args_to_id:
                return self._args_to_id[key]

            idx = len(self._id_to_args)
            self._id_to_args[idx] = (args, kwargs)
            self._args_to_id[key] = idx
            return idx

    def get_args(self, idx: int) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """
        Retrieve args and kwargs from the side table by index.
        """
        # No lock needed - dict reads are atomic in Python
        assert idx in self._id_to_args, f"Index {idx} not found in args side table"
        return self._id_to_args[idx]

    def reset(self) -> None:
        """
        Reset the side table. Only meant for use in unit tests.
        """
        with self._lock:
            self._args_to_id.clear()
            self._id_to_args.clear()

    def __len__(self) -> int:
        return len(self._id_to_args)


# Global singleton for DTensor method arguments (redistribute, to_local, etc.)
dtensor_args_side_table = ArgsSideTable()


def dtensor_to_local(tensor: Any, args_idx: int) -> Any:
    """
    Look up to_local arguments from the side table and call tensor.to_local().

    This function is meant to be used as a call_function target in FX graphs,
    replacing the dynamically-created closure approach.
    """
    args, kwargs = dtensor_args_side_table.get_args(args_idx)
    return tensor.to_local(*args, **kwargs)


# Attach a stable name for debugging and graph readability
dtensor_to_local.__name__ = "dtensor_to_local"
dtensor_to_local.__qualname__ = "dtensor_to_local"


def dtensor_redistribute(tensor: Any, args_idx: int) -> Any:
    """
    Look up redistribute arguments from the side table and call tensor.redistribute().

    This function is meant to be used as a call_function target in FX graphs,
    replacing the dynamically-created closure approach.
    """
    args, kwargs = dtensor_args_side_table.get_args(args_idx)
    return tensor.redistribute(*args, **kwargs)


# Attach a stable name for debugging and graph readability
dtensor_redistribute.__name__ = "dtensor_redistribute"
dtensor_redistribute.__qualname__ = "dtensor_redistribute"


def dtensor_from_local(
    tensor: Any, args_idx: int, shape: Any = None, stride: Any = None
) -> Any:
    """
    Look up from_local arguments from the side table and call DTensor.from_local().

    This function is meant to be used as a call_function target in FX graphs,
    replacing the dynamically-created closure approach.

    The shape and stride parameters are passed separately because they may be
    symbolic and need to be traced through the graph.
    """
    from torch.distributed.tensor import DTensor

    args, kwargs = dtensor_args_side_table.get_args(args_idx)
    return DTensor.from_local(tensor, *args, **kwargs, shape=shape, stride=stride)


# Attach a stable name for debugging and graph readability
dtensor_from_local.__name__ = "dtensor_from_local"
dtensor_from_local.__qualname__ = "dtensor_from_local"


def get_dtensor_args_hash(args_idx: int) -> str:
    """
    Get a hashable representation of the args at the given index.

    This is used for cache key generation, similar to how Triton kernels
    use source code for cache keys.
    """
    args, kwargs = dtensor_args_side_table.get_args(args_idx)
    # Create a string representation that captures the essential properties
    # of the arguments for cache invalidation purposes
    return repr((args, kwargs))


def get_dtensor_args_hashes_from_gm(gm: "torch.fx.GraphModule") -> list[str]:
    """
    Extract all DTensor args hashes from a graph module.

    This function scans the graph for calls to dtensor_to_local,
    dtensor_redistribute, and dtensor_from_local, extracts the args_idx
    from each call, and returns a list of hash strings for cache key generation.

    Similar to get_triton_source_codes_from_gm in autograd_cache.py.
    """
    import torch.fx

    hashes = []
    for module in gm.modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue
        for node in module.graph.nodes:
            if node.op == "call_function" and node.target in (
                dtensor_to_local,
                dtensor_redistribute,
                dtensor_from_local,
            ):
                # The args_idx is passed as a kwarg
                args_idx = node.kwargs.get("args_idx")
                if args_idx is not None:
                    hashes.append(get_dtensor_args_hash(args_idx))
    return hashes

