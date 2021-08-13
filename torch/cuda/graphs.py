import ctypes
import gc
import torch

from ._utils import _dummy_type


if not hasattr(torch._C, '_CudaStreamBase'):
    # Define dummy base classes
    torch._C.__dict__['CUDAGraph'] = _dummy_type('CUDAGraph')
    torch._C.__dict__['graph_pool_handle'] = _dummy_type('graph_pool_handle')

from torch._C import (
    CUDAGraph,
    graph_pool_handle
)


class graph(object):
    r"""Context-manager that captures CUDA work into a :class:`torch.cuda.CUDAGraph`
    object for later replay.

    See :ref:`CUDA Graphs <cuda-graph-semantics>` for a general introduction,
    detailed use, and constraints.

    Arguments:
        graph (torch.cuda.CUDAGraph): Graph object used for capture.
        pool: Opaque token (returned by a call to :func:`~torch.cuda.graph_pool_handle()` or :meth:`other_Graph_instance.pool()<torch.cuda.CUDAGraph.pool`) hinting this graph's capture may share memory from the specified pool.
        stream (torch.cuda.Stream, optional): If supplied, will be set as the current ambient stream in the context.  If not supplied, ``Graph`` sets its own internal side stream as the ambient stream in the context.

    .. note::
        For effective memory sharing, if you pass a ``pool`` used by a previous capture and the previous capture
        used an explicit ``stream`` argument, you should pass the same ``stream`` argument to this capture.

    """
    # default_capture_stream = torch.cuda.Stream()

    def __init__(self,
                 graph,
                 pool=None,
                 stream=None):
        self.capture_stream = torch.cuda.Stream() if stream is None else stream
        self.stream_ctx = torch.cuda.stream
        self.graph = graph

    def __enter__(self):
        # Free as much memory as we can for the graph
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        # Stackoverflow seems comfortable with this pattern
        # https://stackoverflow.com/questions/26635684/calling-enter-and-exit-manually#39172487
        self.stream_ctx.__enter__()

        self.graph.capture_begin()


    def __exit__(self, exc_type, exc_value, traceback):
        self.graph.capture_end()
        self.stream_ctx.__exit__(exc_type, exc_value, traceback)
        # returning None should propagate exceptions from either capture_end or stream_ctx.__exit__()


def make_graphed_callables(callables,
                           sample_inputs,
                           autograd_aware=True,
                           stream=None):
    r"""
    Accepts callables (functions or :class:`nn.Module<torch.nn.Module>`\ s)
    and returns graphed versions.

    Each graphed callable's forward pass runs its source callable's
    forward CUDA work as a CUDA graph inside a single autograd node.

    If ``autograd_aware`` is True, the graphed callable's forward pass also appends
    a backward node to the autograd graph. During backward, this node runs the
    callable's backward work as a CUDA graph.

    Therefore, each graphed callable should be a drop-in replacement for its source callable
    in an autograd-enabled training loop.

    See :ref:`Partial-network capture<partial-network-capture>` for detailed use and constraints.

    .. note::
        The ``requires_grad`` state of each Tensor in ``sample_inputs`` must match the state
        that's expected for the corresponding real input in the training loop.
    """
    pass
