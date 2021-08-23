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
        cuda_graph (torch.cuda.CUDAGraph): Graph object used for capture.
        pool (optional): Opaque token (returned by a call to :func:`~torch.cuda.graph_pool_handle()` or
            :meth:`other_Graph_instance.pool()<torch.cuda.CUDAGraph.pool>`) hinting this graph's capture
            may share memory from the specified pool. See :ref:`Graph memory management<graph-memory-management>`.
        stream (torch.cuda.Stream, optional): If supplied, will be set as the current ambient stream in the context.
            If not supplied, ``Graph`` sets its own internal side stream as the ambient stream in the context.

    .. note::
        For effective memory sharing, if you pass a ``pool`` used by a previous capture and the previous capture
        used an explicit ``stream`` argument, you should pass the same ``stream`` argument to this capture.

    """
    default_capture_stream = None

    def __init__(self,
                 cuda_graph,
                 pool=None,
                 stream=None):
        # Lazy-init of default_capture_stream helps avoid circular-import errors.
        # Not thread safe, but graphs already have the general (explicitly documented)
        # restriction that only one capture may be underway at a time in the process.
        if self.__class__.default_capture_stream is None:
            self.__class__.default_capture_stream = torch.cuda.Stream()

        self.pool = () if pool is None else (pool,)
        self.capture_stream = torch.cuda.Stream() if stream is None else self.__class__.default_capture_stream
        assert self.capture_stream is not None
        self.stream_ctx = torch.cuda.stream(self.capture_stream)
        self.cuda_graph = cuda_graph

    def __enter__(self):
        # Free as much memory as we can for the graph
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        # Stackoverflow seems comfortable with this pattern
        # https://stackoverflow.com/questions/26635684/calling-enter-and-exit-manually#39172487
        self.stream_ctx.__enter__()

        self.cuda_graph.capture_begin(*self.pool)


    def __exit__(self, exc_type, exc_value, traceback):
        self.cuda_graph.capture_end()
        self.stream_ctx.__exit__(exc_type, exc_value, traceback)
        # returning None should propagate exceptions from either capture_end or stream_ctx.__exit__()


def make_graphed_callables(callables,
                           sample_inputs,
                           autograd_aware=True):
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

    If you pass a tuple of several callables, their captures will use the same memory pool.
    See :ref:`Graph memory management<graph-memory-management>` for when this is appropriate.

    Arguments:
        callables (torch.nn.Module or Python function, or tuple of these): Callable or callables to graph.
            See :ref:`Graph memory management<graph-memory-management>` for when passing a tuple of callables
            is appropriate.  If you pass a tuple of callables, their order in the tuple must be the same order
            they'll run in the live workload.
        sample_inputs (tuple or tuple of tuples): Samples inputs for each callable. If a single callable
            was passed, ``sample_inputs`` must be a single tuple. If a tuple of callables was passed,
            ``sample_inputs`` must be tuple of argument tuples.
        autograd_aware (bool, optional, default=True): If True, returned callables will have graphed backward
            as well as forward passes, and can be used in training loops.

    .. note::
        The ``requires_grad`` state of each Tensor in ``sample_inputs`` must match the state
        that's expected for the corresponding real input in the training loop.
    """
    # DO NOT REVIEW BELOW THIS LINE YET

    if not isinstance(callables, tuple):
        callables = (callables,)
        sample_inputs = (sample_inputs,)

    # If a callable is an nn.Module, its graph's full input surface is the args the user explicitly
    # passes to forward (ie, its sample_inputs) AND the module's parameter attributes.
    per_callable_module_params = [tuple(module.parameters())
                                  if isinstance(c, torch.nn.Module) else () for c in callables]
    full_input_surfaces = [sample_args[i] + module_params[i] for i in range(len(callables))]

    fwd_graphs = [torch.cuda.CUDAGraph() for _ in range(len(callables))]
    bwd_graphs = [torch.cuda.CUDAGraph() for _ in range(len(callables))]

    mempool = torch.cuda.graph_pool_handle()
    with torch.cuda.stream(stream):
        per_callable_static_outputs = []
        for i, func, fwd_graph in zip(callables, fwd_graphs, sample_inputs):
            with torch.cuda.graph(fwd_graph):
                static_outputs = func(*sample_args)

            # For simplicity, assumes model output is a tensor or tuple of tensors
            if isinstance(static_outputs, torch.Tensor):
                outputs_was_tensor = True
                static_outputs = (static_outputs,)

        # Most of the spaghetti here comes from handling args that may not require grad (eg data)
        args_require_grad = tuple(i for i in functional_args if i.requires_grad)

        # For simplicity the following assumes all static_outputs require grad, but accommodating some that
        # don't can also be handled with some filtered/padded lists.
        static_incoming_grads = tuple(torch.empty_like(o) for o in static_outputs)

        # Capture gradient creation
        bwd_graph = torch.cuda.CUDAGraph()
        # fwd_graph and bwd_graph will be replayed sequentially, so it's fine for them to share a mempool
        bwd_graph.capture_begin(pool=mempool)
        # grad_inputs = tuple(torch.zeros_like(o) for o in args_require_grad)
        grad_inputs = torch.autograd.grad(outputs=static_outputs,
                                          inputs=args_require_grad,
                                          grad_outputs=static_incoming_grads,
                                          only_inputs=True,
                                          allow_unused=False)
        bwd_graph.capture_end()

        static_inputs = tuple(i.detach() for i in functional_args)
        # not sure if it's necessary to manually reset requires_grad_ on outputs but it doesn't do any harm
        static_outputs = tuple(o.detach().requires_grad_(o.requires_grad) for o in static_outputs)

        # Constructs a list suitable for returning from Graphed.backward:
        # Pads out the actually-needed grads with Nones in gradient slots for inputs that don't require grad.
        # I couldn't think of a slick one-liner for this pattern.
        static_grad_inputs = []
        grad_idx = 0
        for arg in functional_args:
            if arg.requires_grad:
                static_grad_inputs.append(grad_inputs[grad_idx])
                grad_idx += 1
            else:
                static_grad_inputs.append(None)
        static_grad_inputs = tuple(static_grad_inputs)

        class Graphed(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *inputs):
                with torch.no_grad():
                    for i, arg in zip(static_inputs, inputs):
                        if i.data_ptr() != arg.data_ptr():
                            i.copy_(arg)
                fwd_graph.replay()
                return static_outputs

            @staticmethod
            def backward(ctx, *grads):
                with torch.no_grad():
                    for g, grad in zip(static_incoming_grads, grads):
                        if g is None:
                            assert grad is None
                        else:
                            # don't copy if autograd gods have been kind and the
                            # incoming grad is already in the right place
                            if g.data_ptr() != grad.data_ptr():
                                g.copy_(grad)
                bwd_graph.replay()

                # Input args that didn't require grad expect a None gradient.
                return tuple(b.detach() if b is not None else b for b in static_grad_inputs)

        def functionalized(self, *user_args):
            out = Graphed.apply(*(user_args + module_params))
            return out[0] if outputs_was_tensor else out

        if isinstance(func, nn.Module)
            module.forward = types.MethodType(functionalized, module)

    return module
