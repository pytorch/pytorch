r"""Annotate kernels captured in CUDA graphs for profiler traces.

During CUDA graph capture, :func:`mark_kernels` tags the GPU work nodes
(kernels, memcpys, memsets) captured within its scope with user metadata.
The recorded annotations are keyed by the same ``graph node id`` values
that CUPTI-based profilers attach to kernel events during graph replay, so
the annotations can later be joined against a profiler trace to identify
which kernels belong to which region of the captured workload.

Annotation recording is enabled per capture via the ``enable_annotations``
argument of :class:`torch.cuda.graph`::

    import torch
    from torch.cuda.graph_annotations import mark_kernels, get_kernel_annotations

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, enable_annotations=True):
        with mark_kernels("phase_A"):
            y = workload_a(x)
        with mark_kernels({"name": "phase_B", "dtype": "bf16"}):
            z = workload_b(y)

    annotations = get_kernel_annotations()

The annotation mapping is typically pickled and joined against a profiler
trace offline (matching each kernel event's ``graph node id`` field).

Requires the ``cuda-bindings`` package and a CUDA driver that supports
``cudaGraphNodeGetToolsId`` (CUDA >= 13.1, or an equivalent cuda-compat
package). When unavailable, recording silently degrades to a no-op; use
:func:`is_available` to check support programmatically.

.. warning::
    This API is in prototype and may change in future releases.
"""

from torch.cuda._graph_annotations import (
    clear_kernel_annotations,
    get_kernel_annotations,
    is_available,
    mark_kernels,
)


__all__ = [
    "clear_kernel_annotations",
    "get_kernel_annotations",
    "is_available",
    "mark_kernels",
]

# The implementation lives in the private module (which predates this one and
# has external users); re-export and claim the names so they present as
# torch.cuda.graph_annotations APIs (test_correct_module_names requires it).
for _name in __all__:
    globals()[_name].__module__ = "torch.cuda.graph_annotations"
del _name
