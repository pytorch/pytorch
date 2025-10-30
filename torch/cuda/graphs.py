from __future__ import annotations

import gc
import typing
from collections.abc import Callable
from typing import Optional, overload, TYPE_CHECKING, TypeAlias, Union
from typing_extensions import ParamSpec, Self, TypeVar

import torch
from torch import Tensor


if TYPE_CHECKING:
    # importing _POOL_HANDLE at runtime toplevel causes an import cycle
    from torch.cuda import _POOL_HANDLE

from .._utils import _dummy_type


__all__ = [
    "is_current_stream_capturing",
    "graph_pool_handle",
    "CUDAGraph",
    "graph",
    "make_graphed_callables",
]


_R = TypeVar("_R")
_P = ParamSpec("_P")


if not hasattr(torch._C, "_CudaStreamBase"):
    # Define dummy base classes
    torch._C.__dict__["_CUDAGraph"] = _dummy_type("_CUDAGraph")
    torch._C.__dict__["_graph_pool_handle"] = _dummy_type("_graph_pool_handle")
    torch._C.__dict__["_cuda_isCurrentStreamCapturing"] = _dummy_type(
        "_cuda_isCurrentStreamCapturing"
    )

from torch._C import (  # noqa: F401
    _cuda_isCurrentStreamCapturing,
    _CUDAGraph,
    _graph_pool_handle,
)


def is_current_stream_capturing() -> bool:
    r"""Return True if CUDA graph capture is underway on the current CUDA stream, False otherwise.

    If a CUDA context does not exist on the current device, returns False without initializing the context.
    """
    return _cuda_isCurrentStreamCapturing()


# Python shim helps Sphinx process docstrings more reliably.
def graph_pool_handle() -> _POOL_HANDLE:
    r"""Return an opaque token representing the id of a graph memory pool.

    See :ref:`Graph memory management<graph-memory-management>`.

    .. warning::
        This API is in beta and may change in future releases.
    """
    return torch.cuda._POOL_HANDLE(_graph_pool_handle())


# Python shim helps Sphinx process docstrings more reliably.
class CUDAGraph(torch._C._CUDAGraph):
    r"""Wrapper around a CUDA graph.

    Arguments:
        keep_graph (bool, optional): If ``keep_graph=False``, the
            cudaGraphExec_t will be instantiated on GPU at the end of
            ``capture_end`` and the underlying cudaGraph_t will be
            destroyed. Users who want to query or otherwise modify the
            underlying cudaGraph_t before instantiation can set
            ``keep_graph=True`` and access it via ``raw_cuda_graph`` after
            ``capture_end``. Note that the cudaGraphExec_t will not be
            instantiated at the end of ``capture_end`` in this
            case. Instead, it will be instantiated via an explicit called
            to ``instantiate`` or automatically on the first call to
            ``replay`` if ``instantiate`` was not already called. Calling
            ``instantiate`` manually before ``replay`` is recommended to
            prevent increased latency on the first call to ``replay``. It
            is allowed to modify the raw cudaGraph_t after first calling
            ``instantiate``, but the user must call ``instantiate`` again
            manually to make sure the instantiated graph has these
            changes. Pytorch has no means of tracking these changes.

    .. warning::
        This API is in beta and may change in future releases.

    """

    def __new__(cls, keep_graph: bool = False) -> Self:
        return super().__new__(cls, keep_graph)

    def capture_begin(
        self, pool: Optional[_POOL_HANDLE] = None, capture_error_mode: str = "global"
    ) -> None:
        r"""Begin capturing CUDA work on the current stream.

        Typically, you shouldn't call ``capture_begin`` yourself.
        Use :class:`~torch.cuda.graph` or :func:`~torch.cuda.make_graphed_callables`,
        which call ``capture_begin`` internally.

        Arguments:
            pool (optional): Token (returned by :func:`~torch.cuda.graph_pool_handle` or
                :meth:`other_Graph_instance.pool()<torch.cuda.CUDAGraph.pool>`) that hints this graph may share memory
                with the indicated pool.  See :ref:`Graph memory management<graph-memory-management>`.
            capture_error_mode (str, optional): specifies the cudaStreamCaptureMode for the graph capture stream.
                Can be "global", "thread_local" or "relaxed". During cuda graph capture, some actions, such as cudaMalloc,
                may be unsafe. "global" will error on actions in other threads, "thread_local" will only error for
                actions in the current thread, and "relaxed" will not error on these actions. Do NOT change this setting
                unless you're familiar with `cudaStreamCaptureMode <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g9d0535d93a214cbf126835257b16ba85>`_
        """  # noqa: B950
        super().capture_begin(pool=pool, capture_error_mode=capture_error_mode)

    def capture_end(self) -> None:
        r"""End CUDA graph capture on the current stream.

        After ``capture_end``, ``replay`` may be called on this instance.

        Typically, you shouldn't call ``capture_end`` yourself.
        Use :class:`~torch.cuda.graph` or :func:`~torch.cuda.make_graphed_callables`,
        which call ``capture_end`` internally.
        """
        super().capture_end()

    def instantiate(self) -> None:
        r"""Instantiate the CUDA graph. Will be called by
        ``capture_end`` if ``keep_graph=False``, or by ``replay`` if
        ``keep_graph=True`` and ``instantiate`` has not already been
        explicitly called. Does not destroy the cudaGraph_t returned
        by ``raw_cuda_graph``.
        """
        super().instantiate()

    def replay(self) -> None:
        r"""Replay the CUDA work captured by this graph."""
        super().replay()

    def reset(self) -> None:
        r"""Delete the graph currently held by this instance."""
        super().reset()

    def pool(self) -> _POOL_HANDLE:
        r"""Return an opaque token representing the id of this graph's memory pool.

        This id can optionally be passed to another graph's ``capture_begin``,
        which hints the other graph may share the same memory pool.
        """
        return super().pool()

    def enable_debug_mode(self) -> None:
        r"""Enable debugging mode for CUDAGraph.debug_dump."""
        return super().enable_debug_mode()

    def debug_dump(self, debug_path: str) -> None:
        r"""
        Arguments:
            debug_path (required): Path to dump the graph to.

        Calls a debugging function to dump the graph if the debugging is
        enabled via CUDAGraph.enable_debug_mode()
        """
        return super().debug_dump(debug_path)

    def raw_cuda_graph(self) -> int:
        r"""Returns the underlying cudaGraph_t. ``keep_graph`` must be True.

        See the following for APIs for how to manipulate this object: `Graph Managmement <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html>`_ and `cuda-python Graph Management bindings <https://nvidia.github.io/cuda-python/cuda-bindings/latest/module/runtime.html#graph-management>`_
        """  # noqa: B950
        return super().raw_cuda_graph()

    def raw_cuda_graph_exec(self) -> int:
        r"""Returns the underlying cudaGraphExec_t. ``instantiate`` must have been called if ``keep_graph`` is True, or ``capture_end`` must have been called if ``keep_graph`` is False. If you call ``instantiate()`` after ``raw_cuda_graph_exec()``, the previously returned cudaGraphExec_t will be destroyed. It is your responsibility not to use this object after destruction.

        See the following for APIs for how to manipulate this object: `Graph Execution <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH__EXEC.html>`_ and `cuda-python Graph Execution bindings <https://nvidia.github.io/cuda-python/cuda-bindings/latest/module/runtime.html#graph-execution>`_
        """  # noqa: B950
        return super().raw_cuda_graph_exec()


class graph:
    r"""Context-manager that captures CUDA work into a :class:`torch.cuda.CUDAGraph` object for later replay.

    See :ref:`CUDA Graphs <cuda-graph-semantics>` for a general introduction,
    detailed use, and constraints.

    Arguments:
        cuda_graph (torch.cuda.CUDAGraph): Graph object used for capture.
        pool (optional): Opaque token (returned by a call to :func:`~torch.cuda.graph_pool_handle()` or
            :meth:`other_Graph_instance.pool()<torch.cuda.CUDAGraph.pool>`) hinting this graph's capture
            may share memory from the specified pool. See :ref:`Graph memory management<graph-memory-management>`.
        stream (torch.cuda.Stream, optional): If supplied, will be set as the current stream in the context.
            If not supplied, ``graph`` sets its own internal side stream as the current stream in the context.
        capture_error_mode (str, optional): specifies the cudaStreamCaptureMode for the graph capture stream.
            Can be "global", "thread_local" or "relaxed". During cuda graph capture, some actions, such as cudaMalloc,
            may be unsafe. "global" will error on actions in other threads, "thread_local" will only error for
            actions in the current thread, and "relaxed" will not error on actions. Do NOT change this setting
            unless you're familiar with `cudaStreamCaptureMode <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g9d0535d93a214cbf126835257b16ba85>`_

    .. note::
        For effective memory sharing, if you pass a ``pool`` used by a previous capture and the previous capture
        used an explicit ``stream`` argument, you should pass the same ``stream`` argument to this capture.

    .. warning::
        This API is in beta and may change in future releases.

    .. _cudaStreamCaptureMode:
        https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g9d0535d93a214cbf126835257b16ba85
    """  # noqa: B950

    default_capture_stream: Optional[torch.cuda.Stream] = None

    def __init__(
        self,
        cuda_graph: CUDAGraph,
        pool: Optional[_POOL_HANDLE] = None,
        stream: Optional[torch.cuda.Stream] = None,
        capture_error_mode: str = "global",
    ):
        # Lazy-init of default_capture_stream helps avoid circular-import errors.
        # Not thread safe, but graphs already have the general (explicitly documented)
        # restriction that only one capture may be underway at a time in the process.
        if self.__class__.default_capture_stream is None:
            self.__class__.default_capture_stream = torch.cuda.Stream()

        self.pool: Union[tuple[()], tuple[_POOL_HANDLE]] = (
            () if pool is None else (pool,)
        )
        self.capture_stream = (
            stream if stream is not None else self.__class__.default_capture_stream
        )
        assert self.capture_stream is not None
        self.stream_ctx = torch.cuda.stream(self.capture_stream)
        self.cuda_graph = cuda_graph
        self.capture_error_mode = capture_error_mode

    def __enter__(self) -> None:
        # Free as much memory as we can for the graph
        torch.cuda.synchronize()

        if torch.compiler.config.force_cudagraph_gc:
            # Originally we unconditionally garbage collected here. On one hand
            # that's nice because we have a chance to collect more memory, but
            # on the other hand it is REALLY expensive, especially for doing
            # multiple cudagraph captures in a row. In theory it will only help
            # when a dead python cycle is holding onto CUDA memory.
            gc.collect()

        torch.cuda.empty_cache()

        # Stackoverflow seems comfortable with this pattern
        # https://stackoverflow.com/questions/26635684/calling-enter-and-exit-manually#39172487
        self.stream_ctx.__enter__()

        self.cuda_graph.capture_begin(
            # type: ignore[misc]
            *self.pool,
            # pyrefly: ignore [bad-keyword-argument]
            capture_error_mode=self.capture_error_mode,
        )

    def __exit__(self, *args: object) -> None:
        self.cuda_graph.capture_end()
        self.stream_ctx.__exit__(*args)
        # returning None should propagate exceptions from either capture_end or stream_ctx.__exit__()


_ModuleOrCallable: TypeAlias = Union["torch.nn.Module", Callable[..., object]]


@overload
def make_graphed_callables(
    callables: _ModuleOrCallable,
    sample_args: tuple[Tensor, ...],
    num_warmup_iters: int = 3,
    allow_unused_input: bool = False,
    pool: Optional[_POOL_HANDLE] = None,
) -> _ModuleOrCallable: ...


@overload
def make_graphed_callables(
    callables: tuple[_ModuleOrCallable, ...],
    sample_args: tuple[tuple[Tensor, ...], ...],
    num_warmup_iters: int = 3,
    allow_unused_input: bool = False,
    pool: Optional[_POOL_HANDLE] = None,
) -> tuple[_ModuleOrCallable, ...]: ...


def make_graphed_callables(
    callables: Union[_ModuleOrCallable, tuple[_ModuleOrCallable, ...]],
    sample_args: Union[tuple[Tensor, ...], tuple[tuple[Tensor, ...], ...]],
    num_warmup_iters: int = 3,
    allow_unused_input: bool = False,
    pool: Optional[_POOL_HANDLE] = None,
) -> Union[_ModuleOrCallable, tuple[_ModuleOrCallable, ...]]:
    r"""Accept callables (functions or :class:`nn.Module<torch.nn.Module>`\ s) and returns graphed versions.

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
        num_warmup_iters (int): The number of warmup iterations. Currently, ``DataDistributedParallel`` needs
            11 iterations for warm up. Default: ``3``.
        allow_unused_input (bool): If False, specifying inputs that were not used when computing outputs
            (and therefore their grad is always zero) is an error. Defaults to False.
        pool (optional): Token (returned by :func:`~torch.cuda.graph_pool_handle` or
            :meth:`other_Graph_instance.pool()<torch.cuda.CUDAGraph.pool>`) that hints this graph may share memory
            with the indicated pool.  See :ref:`Graph memory management<graph-memory-management>`.
    .. note::
        The ``requires_grad`` state of each Tensor in ``sample_args`` must match the state
        that's expected for the corresponding real input in the training loop.

    .. warning::
        This API is in beta and may change in future releases.

    .. warning::
        ``sample_args`` for each callable must contain only Tensors. Other types are not allowed.

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
        The automatic mixed precision is supported in :func:`~torch.cuda.make_graphed_callables` only with disabled
        caching. The context manager `torch.cuda.amp.autocast()` must have `cache_enabled=False`.
    """
    if torch.is_autocast_enabled() and torch.is_autocast_cache_enabled():
        raise RuntimeError(
            "make_graphed_callables does not support the autocast caching. Please set `cache_enabled=False`."
        )

    just_one_callable = False

    _sample_args: tuple[tuple[Tensor, ...], ...]
    if not isinstance(callables, tuple):
        just_one_callable = True
        callables = (callables,)
        _sample_args = (typing.cast(tuple[Tensor, ...], sample_args),)
    else:
        _sample_args = typing.cast(tuple[tuple[Tensor, ...], ...], sample_args)

    flatten_sample_args = []

    for c, args in zip(callables, _sample_args):
        if isinstance(c, torch.nn.Module):
            assert (
                len(c._backward_hooks) == 0
                and len(c._forward_hooks) == 0
                and len(c._forward_pre_hooks) == 0
            ), (
                "Modules must not have hooks registered at the time they are passed. However, registering hooks "
                + "on modules after passing them through make_graphed_callables is allowed."
            )
            assert all(b.requires_grad is False for b in c.buffers()), (
                "In any :class:`~torch.nn.Module` passed to "
                + ":func:`~make_graphed_callables`, only parameters may be trainable. All buffers must have "
                + "``requires_grad=False``."
            )
        flatten_arg = torch.utils._pytree.arg_tree_leaves(*args)
        flatten_sample_args.append(tuple(flatten_arg))
        assert all(isinstance(arg, torch.Tensor) for arg in flatten_arg), (
            "In the beta API, sample_args "
            + "for each callable must contain only Tensors. Other types are not allowed."
        )

    # If a callable is an nn.Module, its graph's full input surface is the args the user explicitly
    # passes to forward (ie, its sample_args) AND the module's parameter attributes.
    per_callable_len_user_args = [len(args) for args in flatten_sample_args]
    per_callable_module_params = [
        tuple(c.parameters()) if isinstance(c, torch.nn.Module) else ()
        for c in callables
    ]
    per_callable_static_input_surfaces = [
        flatten_sample_args[i] + per_callable_module_params[i]
        for i in range(len(callables))
    ]

    fwd_graphs = [torch.cuda.CUDAGraph() for _ in range(len(callables))]
    bwd_graphs = [torch.cuda.CUDAGraph() for _ in range(len(callables))]

    mempool = graph_pool_handle() if pool is None else pool

    # Warmup
    # Hopefully prevents cudnn benchmarking and other lazy-initialization cuda work
    # from ending up in any captures.
    torch.cuda.synchronize()
    with torch.cuda.stream(torch.cuda.Stream()):
        for func, args, static_input_surface in zip(
            callables, _sample_args, per_callable_static_input_surfaces
        ):
            grad_inputs, outputs, outputs_grad = None, None, None
            for _ in range(num_warmup_iters):
                outputs = torch.utils._pytree.tree_leaves(func(*args))
                outputs_grad = tuple(o for o in outputs if o.requires_grad)
                if len(outputs_grad) > 0:
                    grad_inputs = torch.autograd.grad(
                        outputs=outputs_grad,
                        inputs=tuple(
                            i for i in static_input_surface if i.requires_grad
                        ),
                        grad_outputs=tuple(
                            torch.empty_like(o) for o in outputs if o.requires_grad
                        ),
                        only_inputs=True,
                        allow_unused=allow_unused_input,
                    )
            for v in [outputs, outputs_grad, grad_inputs]:
                del v

    torch.cuda.synchronize()

    # All captures here share a mempool. To avoid replays corrupting each other's memory,
    # the safest approach is to capture all passes in the same order they'll run:
    # fwd 1, fwd 2, ... fwd N, then bwd N, bwd N-1, ... bwd 1.

    # Capture forward graphs
    per_callable_static_outputs = []
    per_callable_output_unflatten_spec = []
    for func, args, fwd_graph in zip(callables, _sample_args, fwd_graphs):
        with torch.cuda.graph(fwd_graph, pool=mempool):
            func_outputs = func(*args)

        flatten_outputs, spec = torch.utils._pytree.tree_flatten(func_outputs)
        per_callable_static_outputs.append(tuple(flatten_outputs))
        per_callable_output_unflatten_spec.append(spec)

    # Capture backward graphs in reverse order
    per_callable_static_grad_outputs = []
    per_callable_static_grad_inputs = []
    for static_input_surface, static_outputs, bwd_graph in zip(
        reversed(per_callable_static_input_surfaces),
        reversed(per_callable_static_outputs),
        reversed(bwd_graphs),
    ):
        # For now, assumes all static_outputs require grad
        # assert all(o.requires_grad for o in static_outputs), "Outputs of graphed callables must require grad."
        static_grad_outputs = tuple(
            torch.empty_like(o) if o.requires_grad else None for o in static_outputs
        )

        outputs_grad = tuple(o for o in static_outputs if o.requires_grad)
        grad_inputs = None
        if len(outputs_grad) > 0:
            with torch.cuda.graph(bwd_graph, pool=mempool):
                grad_inputs = torch.autograd.grad(
                    outputs=outputs_grad,
                    inputs=tuple(i for i in static_input_surface if i.requires_grad),
                    grad_outputs=tuple(o for o in static_grad_outputs if o is not None),
                    only_inputs=True,
                    allow_unused=allow_unused_input,
                )

        # Constructs a tuple suitable for returning from Graphed.backward:
        # Pads out the actually-needed grads with Nones in gradient slots for inputs that don't require grad.
        # I couldn't think of a slick one-liner for this pattern.
        static_grad_inputs = []
        grad_idx = 0
        for arg in static_input_surface:
            if arg.requires_grad and grad_inputs is not None:
                static_grad_inputs.append(grad_inputs[grad_idx])
                grad_idx += 1
            else:
                static_grad_inputs.append(None)  # type: ignore[arg-type]
        static_grad_inputs = tuple(static_grad_inputs)  # type: ignore[assignment]

        per_callable_static_grad_outputs.append(static_grad_outputs)
        per_callable_static_grad_inputs.append(static_grad_inputs)

    # Reverses the most recent two lists
    per_callable_static_grad_outputs.reverse()
    per_callable_static_grad_inputs.reverse()
    # Now for every per_callable list, per_callable_*[i] holds the stuff for the ith callable.

    def make_graphed_autograd_function(
        fwd_graph: CUDAGraph,
        bwd_graph: CUDAGraph,
        module_params: tuple[torch.nn.Parameter, ...],
        len_user_args: int,
        output_unflatten_spec: torch.utils._pytree.TreeSpec,
        static_input_surface: tuple[Tensor, ...],
        static_outputs: tuple[Tensor, ...],
        static_grad_outputs: tuple[Optional[Tensor], ...],
        static_grad_inputs: tuple[Tensor, ...],
    ) -> Callable[..., object]:
        class Graphed(torch.autograd.Function):
            @staticmethod
            # pyrefly: ignore [bad-override]
            def forward(ctx: object, *inputs: Tensor) -> tuple[Tensor, ...]:
                # At this stage, only the user args may (potentially) be new tensors.
                for i in range(len_user_args):
                    if static_input_surface[i].data_ptr() != inputs[i].data_ptr():
                        static_input_surface[i].copy_(inputs[i])
                fwd_graph.replay()
                assert isinstance(static_outputs, tuple)
                return tuple(o.detach() for o in static_outputs)

            @staticmethod
            @torch.autograd.function.once_differentiable
            # pyrefly: ignore [bad-override]
            def backward(ctx: object, *grads: Tensor) -> tuple[Tensor, ...]:
                assert len(grads) == len(static_grad_outputs)
                for g, grad in zip(static_grad_outputs, grads):
                    if g is not None:
                        # don't copy if autograd gods have been kind and the
                        # incoming grad is already in the right place
                        if g.data_ptr() != grad.data_ptr():
                            g.copy_(grad)
                bwd_graph.replay()

                # Input args that didn't require grad expect a None gradient.
                assert isinstance(static_grad_inputs, tuple)
                return tuple(
                    # pyrefly: ignore [bad-argument-type]
                    b.detach() if b is not None else b
                    for b in static_grad_inputs
                )

        def functionalized(*user_args: object) -> object:
            # Runs the autograd function with inputs == all inputs to the graph that might require grad
            # (explicit user args + module parameters)
            # Assumes module params didn't change since capture.
            flatten_user_args = torch.utils._pytree.arg_tree_leaves(*user_args)
            out = Graphed.apply(*(tuple(flatten_user_args) + module_params))
            return torch.utils._pytree.tree_unflatten(out, output_unflatten_spec)

        return functionalized

    # Put together the final graphed callables
    ret: list[_ModuleOrCallable] = []
    for i, func in enumerate(callables):
        graphed = make_graphed_autograd_function(
            fwd_graphs[i],
            bwd_graphs[i],
            per_callable_module_params[i],
            per_callable_len_user_args[i],
            per_callable_output_unflatten_spec[i],
            per_callable_static_input_surfaces[i],
            per_callable_static_outputs[i],
            per_callable_static_grad_outputs[i],
            per_callable_static_grad_inputs[i],
        )

        if isinstance(func, torch.nn.Module):

            def make_graphed_forward(
                func: torch.nn.Module,
                graph_training_state: bool,
                graphed: Callable[_P, _R],
                orig_fwd: Callable[_P, _R],
            ) -> Callable[_P, _R]:
                def new_fwd(*user_args: _P.args, **user_kwargs: _P.kwargs) -> _R:
                    # If the module's training-or-eval state matches what we graphed,
                    # run the graph, otherwise run the original forward method
                    if func.training == graph_training_state:
                        return graphed(*user_args, **user_kwargs)
                    else:
                        return orig_fwd(*user_args, **user_kwargs)

                return new_fwd

            func.forward = make_graphed_forward(
                func, func.training, graphed, func.forward
            )
            ret.append(func)
        else:
            ret.append(graphed)

    if just_one_callable:
        return ret[0]

    return tuple(ret)
