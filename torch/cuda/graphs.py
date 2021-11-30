import gc
import torch

from ._utils import _dummy_type


if not hasattr(torch._C, '_CudaStreamBase'):
    # Define dummy base classes
    torch._C.__dict__['_CUDAGraph'] = _dummy_type('_CUDAGraph')
    torch._C.__dict__['_graph_pool_handle'] = _dummy_type('_graph_pool_handle')

from torch._C import _CUDAGraph  # noqa: F401
from torch._C import _graph_pool_handle


# Python shim helps Sphinx process docstrings more reliably.
def graph_pool_handle():
    r"""
    Returns an opaque token representing the id of a graph memory pool.
    See :ref:`Graph memory management<graph-memory-management>`.

    .. warning::
        This API is in beta and may change in future releases.
    """
    return _graph_pool_handle()


# Python shim helps Sphinx process docstrings more reliably.
class CUDAGraph(torch._C._CUDAGraph):
    r"""
    Wrapper around a CUDA graph.

    .. warning::
        This API is in beta and may change in future releases.
    """
    def __new__(cls):
        return super(CUDAGraph, cls).__new__(cls)

    def __init__(self):
        super(CUDAGraph, self).__init__()

    def capture_begin(self, pool=None):
        r"""
        Begins capturing CUDA work on the current stream.

        Typically, you shouldn't call ``capture_begin`` yourself.
        Use :class:`~torch.cuda.graph` or :func:`~torch.cuda.make_graphed_callables`,
        which call ``capture_begin`` internally.

        Arguments:
            pool (optional): Token (returned by :func:`~torch.cuda.graph_pool_handle` or
                :meth:`other_Graph_instance.pool()<torch.cuda.CUDAGraph.pool>`) that hints this graph may share memory
                with the indicated pool.  See :ref:`Graph memory management<graph-memory-management>`.
        """
        # I'm not sure if pybind11 converts a None arg to the default defined on the C++ side,
        # so I'm not taking any chances.
        if pool is None:
            super(CUDAGraph, self).capture_begin()
        else:
            super(CUDAGraph, self).capture_begin(pool)

    def capture_end(self):
        r"""
        Ends CUDA graph capture on the current stream.
        After ``capture_end``, ``replay`` may be called on this instance.

        Typically, you shouldn't call ``capture_end`` yourself.
        Use :class:`~torch.cuda.graph` or :func:`~torch.cuda.make_graphed_callables`,
        which call ``capture_end`` internally.
        """
        super(CUDAGraph, self).capture_end()

    def replay(self):
        r"""
        Replays the CUDA work captured by this graph.
        """
        super(CUDAGraph, self).replay()

    def reset(self):
        r"""
        Deletes the graph currently held by this instance.
        """
        super(CUDAGraph, self).reset()

    def pool(self):
        r"""
        Returns an opaque token representing the id of this graph's memory pool.
        This id can optionally be passed to another graph's ``capture_begin``,
        which hints the other graph may share the same memory pool.
        """
        return super(CUDAGraph, self).pool()


class graph(object):
    r"""
    Context-manager that captures CUDA work into a :class:`torch.cuda.CUDAGraph`
    object for later replay.

    See :ref:`CUDA Graphs <cuda-graph-semantics>` for a general introduction,
    detailed use, and constraints.

    Arguments:
        cuda_graph (torch.cuda.CUDAGraph): Graph object used for capture.
        pool (optional): Opaque token (returned by a call to :func:`~torch.cuda.graph_pool_handle()` or
            :meth:`other_Graph_instance.pool()<torch.cuda.CUDAGraph.pool>`) hinting this graph's capture
            may share memory from the specified pool. See :ref:`Graph memory management<graph-memory-management>`.
        stream (torch.cuda.Stream, optional): If supplied, will be set as the current stream in the context.
            If not supplied, ``graph`` sets its own internal side stream as the current stream in the context.

    .. note::
        For effective memory sharing, if you pass a ``pool`` used by a previous capture and the previous capture
        used an explicit ``stream`` argument, you should pass the same ``stream`` argument to this capture.

    .. warning::
        This API is in beta and may change in future releases.
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
        self.capture_stream = stream if stream is not None else self.__class__.default_capture_stream
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


def make_graphed_callables(callables, sample_args):
    r"""
    Accepts callables (functions or :class:`nn.Module<torch.nn.Module>`\ s)
    and returns graphed versions.

    Each graphed callable's forward pass runs its source callable's
    forward CUDA work as a CUDA graph inside a single autograd node.

    The graphed callable's forward pass also appends
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
        sample_args (tuple of Tensors, or tuple of tuples of Tensors): Samples args for each callable.
            If a single callable was passed, ``sample_args`` must be a single tuple of argument Tensors.
            If a tuple of callables was passed, ``sample_args`` must be tuple of tuples of argument Tensors.

    .. note::
        The ``requires_grad`` state of each Tensor in ``sample_args`` must match the state
        that's expected for the corresponding real input in the training loop.

    .. warning::
        This API is in beta and may change in future releases.

    .. warning::
        ``sample_args`` for each callable must be a tuple of Tensors. Other types and keyword args
        are not allowed.

    .. warning::
        Returned callables do not support higher order differentiation (e.g., double backward).

    .. warning::
        In any :class:`~torch.nn.Module` passed to :func:`~make_graphed_callables`, only parameters
        may be trainable. Buffers must have ``requires_grad=False``.

    .. warning::
        After you pass a :class:`torch.nn.Module` through :func:`~make_graphed_callables`,
        you may not add or remove any of that Module's parameters or buffers.

    .. warning::
        :class:`torch.nn.Module`\s passed to :func:`~torch.cuda.make_graphed_callables` must not have module hooks
        registered on them at the time they are passed. However, registering hooks on modules *after* passing them
        through :func:`~torch.cuda.make_graphed_callables` is allowed.

    .. warning::
        When running a graphed callable, you must pass its arguments in the same order and format
        they appeared in that callable's ``sample_args``.

    .. warning::
        All Tensor outputs of graphed callables must require grad.
    """
    just_one_callable = False

    if not isinstance(callables, tuple):
        just_one_callable = True
        callables = (callables,)
        sample_args = (sample_args,)

    for c, args in zip(callables, sample_args):
        if isinstance(c, torch.nn.Module):
            assert len(c._backward_hooks) == 0 and len(c._forward_hooks) == 0 and len(c._forward_pre_hooks) == 0, \
                "Modules must not have hooks registered at the time they are passed. However, registering hooks " + \
                "on modules after passing them through make_graphed_callables is allowed."
            assert all(b.requires_grad is False for b in c.buffers()), "In any :class:`~torch.nn.Module` passed to " + \
                ":func:`~make_graphed_callables`, only parameters may be trainable. All buffers must have " + \
                "``requires_grad=False``."
        assert all(isinstance(arg, torch.Tensor) for arg in args), "In the beta API, sample_args " + \
            "for each callable must be a tuple of Tensors. Other types and keyword args are not allowed."


    # If a callable is an nn.Module, its graph's full input surface is the args the user explicitly
    # passes to forward (ie, its sample_args) AND the module's parameter attributes.
    per_callable_len_user_args = [len(args) for args in sample_args]
    per_callable_module_params = [tuple(c.parameters()) if isinstance(c, torch.nn.Module) else ()
                                  for c in callables]
    per_callable_static_input_surfaces = [sample_args[i] + per_callable_module_params[i]
                                          for i in range(len(callables))]

    fwd_graphs = [torch.cuda.CUDAGraph() for _ in range(len(callables))]
    bwd_graphs = [torch.cuda.CUDAGraph() for _ in range(len(callables))]

    mempool = graph_pool_handle()

    # Warmup
    # Hopefully prevents cudnn benchmarking and other lazy-initialization cuda work
    # from ending up in any captures.
    torch.cuda.synchronize()
    with torch.cuda.stream(torch.cuda.Stream()):
        for func, args, static_input_surface in zip(callables,
                                                    sample_args,
                                                    per_callable_static_input_surfaces):
            for _ in range(3):
                outputs = func(*args)
                outputs = (outputs,) if isinstance(outputs, torch.Tensor) else outputs
                grad_inputs = torch.autograd.grad(outputs=outputs,
                                                  inputs=tuple(i for i in static_input_surface if i.requires_grad),
                                                  grad_outputs=tuple(torch.empty_like(o) for o in outputs),
                                                  only_inputs=True,
                                                  allow_unused=False)
            del outputs, grad_inputs
    torch.cuda.synchronize()

    # All captures here share a mempool. To avoid replays corrupting each other's memory,
    # the safest approach is to capture all passes in the same order they'll run:
    # fwd 1, fwd 2, ... fwd N, then bwd N, bwd N-1, ... bwd 1.

    # Capture forward graphs
    per_callable_static_outputs = []
    per_callable_output_was_tensor = []
    for func, args, fwd_graph in zip(callables,
                                     sample_args,
                                     fwd_graphs):
        with torch.cuda.graph(fwd_graph, pool=mempool):
            outputs = func(*args)

        # Assumes model output is a tensor or tuple of tensors
        if isinstance(outputs, torch.Tensor):
            per_callable_output_was_tensor.append(True)
            outputs = (outputs,)
        else:
            per_callable_output_was_tensor.append(False)

        per_callable_static_outputs.append(outputs)

    # Capture backward graphs in reverse order
    per_callable_static_grad_outputs = []
    per_callable_static_grad_inputs = []
    for static_input_surface, static_outputs, bwd_graph, module_params in \
            zip(reversed(per_callable_static_input_surfaces),
                reversed(per_callable_static_outputs),
                reversed(bwd_graphs),
                reversed(per_callable_module_params)):

        # For now, assumes all static_outputs require grad
        assert all(o.requires_grad for o in static_outputs), "Outputs of graphed callables must require grad."
        static_grad_outputs = tuple(torch.empty_like(o) for o in static_outputs)

        with torch.cuda.graph(bwd_graph, pool=mempool):
            grad_inputs = torch.autograd.grad(outputs=static_outputs,
                                              inputs=tuple(i for i in static_input_surface if i.requires_grad),
                                              grad_outputs=static_grad_outputs,
                                              only_inputs=True,
                                              allow_unused=False)

        # Constructs a tuple suitable for returning from Graphed.backward:
        # Pads out the actually-needed grads with Nones in gradient slots for inputs that don't require grad.
        # I couldn't think of a slick one-liner for this pattern.
        static_grad_inputs = []
        grad_idx = 0
        for arg in static_input_surface:
            if arg.requires_grad:
                static_grad_inputs.append(grad_inputs[grad_idx])
                grad_idx += 1
            else:
                static_grad_inputs.append(None)  # type: ignore[arg-type]
        static_grad_inputs = tuple(static_grad_inputs)  # type: ignore[assignment]

        per_callable_static_grad_outputs.append(static_grad_outputs)
        per_callable_static_grad_inputs.append(static_grad_inputs)

    # Reverses the most recent two lists
    per_callable_static_grad_outputs = list(reversed(per_callable_static_grad_outputs))
    per_callable_static_grad_inputs = list(reversed(per_callable_static_grad_inputs))
    # Now for every per_callable list, per_callable_*[i] holds the stuff for the ith callable.

    def make_graphed_autograd_function(fwd_graph,
                                       bwd_graph,
                                       module_params,
                                       len_user_args,
                                       output_was_tensor,
                                       static_input_surface,
                                       static_outputs,
                                       static_grad_outputs,
                                       static_grad_inputs):
        class Graphed(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *inputs):
                # At this stage, only the user args may (potentially) be new tensors.
                for i in range(len_user_args):
                    if static_input_surface[i].data_ptr() != inputs[i].data_ptr():
                        static_input_surface[i].copy_(inputs[i])
                fwd_graph.replay()
                assert isinstance(static_outputs, tuple)
                return tuple(o.detach() for o in static_outputs)

            @staticmethod
            @torch.autograd.function.once_differentiable
            def backward(ctx, *grads):
                for g, grad in zip(static_grad_outputs, grads):
                    if g is None:
                        assert grad is None
                    else:
                        # don't copy if autograd gods have been kind and the
                        # incoming grad is already in the right place
                        if g.data_ptr() != grad.data_ptr():
                            g.copy_(grad)
                bwd_graph.replay()

                # Input args that didn't require grad expect a None gradient.
                assert isinstance(static_grad_inputs, tuple)
                return tuple(b.detach() if b is not None else b for b in static_grad_inputs)

        def functionalized(*user_args):
            # Runs the autograd function with inputs == all inputs to the graph that might require grad
            # (explicit user args + module parameters)
            # Assumes module params didn't change since capture.
            out = Graphed.apply(*(user_args + module_params))
            return out[0] if output_was_tensor else out

        return functionalized

    # Put together the final graphed callables
    ret = []
    for i, func in enumerate(callables):
        graphed = make_graphed_autograd_function(fwd_graphs[i],
                                                 bwd_graphs[i],
                                                 per_callable_module_params[i],
                                                 per_callable_len_user_args[i],
                                                 per_callable_output_was_tensor[i],
                                                 per_callable_static_input_surfaces[i],
                                                 per_callable_static_outputs[i],
                                                 per_callable_static_grad_outputs[i],
                                                 per_callable_static_grad_inputs[i])

        if isinstance(func, torch.nn.Module):
            def make_graphed_forward(func, graph_training_state, graphed, orig_fwd):
                def new_fwd(*user_args):
                    # If the module's training-or-eval state matches what we graphed,
                    # run the graph, otherwise run the original forward method
                    if func.training == graph_training_state:
                        return graphed(*user_args)
                    else:
                        return orig_fwd(*user_args)
                return new_fwd
            func.forward = make_graphed_forward(func, func.training, graphed, func.forward)  # type: ignore[assignment]
            ret.append(func)
        else:
            ret.append(graphed)

    if just_one_callable:
        return ret[0]

    return tuple(ret)
