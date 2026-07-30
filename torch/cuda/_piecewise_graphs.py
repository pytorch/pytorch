"""Forward the piecewise-cuda-graphs public API into ``torch.cuda``.

Piecewise CUDA graphs let a workload run under CUDA graphs while sections that
cannot be captured (e.g. attention) run eagerly, by splitting capture into a
sequence of graph and eager segments. The implementation lives in the standalone
``piecewise_cuda_graphs`` package (shipped in ``torchannex``, also installable on
its own); this module re-exports its public entry points so users reach them as
``torch.cuda.piecewise_graph`` and friends, mirroring the core CUDA graph API.

The package is imported lazily on first access: it imports ``torch``, so pulling
it in while ``torch.cuda`` is still initializing would be an import cycle, and we
do not want to pay its import cost for users who never touch the API. When the
package is not installed the names still resolve, but using one raises a
descriptive ImportError pointing at the install command.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING


__all__ = [
    "CUDAGraphSequence",
    "force_no_graph",
    "no_graph",
    "piecewise_graph",
]


if TYPE_CHECKING:
    from piecewise_cuda_graphs import (  # pyrefly: ignore[missing-import]
        CUDAGraphSequence,
        force_no_graph,
        no_graph,
        piecewise_graph,
    )


def _missing(name: str) -> Any:
    # Placeholder returned when the package is absent: the name still resolves so
    # the public API surface is stable, but using it raises with an actionable
    # install hint (mirrors the _dummy_type raise-on-use pattern in graphs.py).
    def raise_missing(*args: object, **kwargs: object) -> Any:
        raise ImportError(
            f"torch.cuda.{name} requires the piecewise-cuda-graphs package: "
            "install torchannex with `pip install torchannex`, or the standalone "
            "package with `pip install piecewise-cuda-graphs`."
        )

    raise_missing.__name__ = name
    raise_missing.__qualname__ = name
    return raise_missing


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        import piecewise_cuda_graphs  # pyrefly: ignore[missing-import]
    except ImportError:
        return _missing(name)
    return getattr(piecewise_cuda_graphs, name)
