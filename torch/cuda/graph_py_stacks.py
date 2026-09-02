r"""Capture Python launch stacks for CUDA graph nodes.

Enable stack capture through :class:`torch.cuda.graph` and retrieve the result after
the graph has been instantiated::

    import torch
    from torch.cuda.graph_py_stacks import take_stacks

    graph = torch.cuda.CUDAGraph()
    with torch.autograd.grad_mode.set_multithreading_enabled(False):
        with torch.cuda.graph(
            graph,
            enable_annotations=True,
            annotation_config={"capture_py_stacks": True},
        ):
            output = workload(input)

    stacks = take_stacks(graph)

The mapping keys are the same ``graph node id`` values emitted by CUPTI-based profilers.

.. warning::
    Stack capture requires ``cupti-python`` and enables a CUPTI subscriber. If no subscriber
    is already active, this prevents :class:`torch.profiler.profile` from initializing later
    in the same process.

.. warning::
    This API is in prototype and may change in future releases.
"""

from __future__ import annotations

from typing import Any


def take_stacks(graph: Any) -> dict[int, str]:
    r"""take_stacks(graph) -> dict[int, str]

    Return and clear the Python launch stacks recorded for a CUDA graph.

    Keys are exec-graph ``toolsId`` values after the graph has been instantiated.

    Args:
        graph (torch.cuda.CUDAGraph): the graph whose captured stacks are returned.

    Returns:
        dict[int, str]: A mapping from graph-node ``toolsId`` to its Python stack.
    """
    from torch.cuda._graph_py_stacks import take_stacks as _take_stacks

    return _take_stacks(graph)


def clear_stacks(graph: Any) -> None:
    r"""clear_stacks(graph) -> None

    Discard Python launch stacks recorded for a CUDA graph.

    Args:
        graph (torch.cuda.CUDAGraph): the graph whose captured stacks are discarded.
    """
    from torch.cuda._graph_py_stacks import clear_stacks as _clear_stacks

    _clear_stacks(graph)


__all__ = ["clear_stacks", "take_stacks"]
