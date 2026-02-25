from __future__ import annotations

import typing
from collections.abc import Callable
from typing import Optional, overload, TYPE_CHECKING, TypeAlias, Union
from typing_extensions import ParamSpec, Self, TypeVar

import torch
from torch import Tensor


if TYPE_CHECKING:
    from torch.xpu import _POOL_HANDLE

from .._utils import _dummy_type


__all__ = [
    "is_current_stream_capturing",
    "graph_pool_handle",
    "XPUGraph",
    "graph",
    "make_graphed_callables",
]

_R = TypeVar("_R")
_P = ParamSpec("_P")

if not hasattr(torch._C, "_XpuStreamBase"):
    # Define dummy base classes
    torch._C.__dict__["_XPUGraph"] = _dummy_type("_XPUGraph")
    torch._C.__dict__["_xpu_graph_pool_handle"] = _dummy_type("_xpu_graph_pool_handle")
    torch._C.__dict__["_xpu_isCurrentStreamCapturing"] = _dummy_type(
        "_xpu_isCurrentStreamCapturing"
    )

from torch._C import _xpu_graph_pool_handle, _xpu_isCurrentStreamCapturing, _XPUGraph


def is_current_stream_capturing() -> bool:
    r"""Return True if XPU graph capture is underway on the current XPU stream, False otherwise.

    If a XPU context does not exist on the current device, returns False without initializing the context.
    """
    return _xpu_isCurrentStreamCapturing()


def graph_pool_handle() -> _POOL_HANDLE:
    r"""Return an opaque token representing the id of a graph memory pool."""
    return torch.xpu._POOL_HANDLE(_xpu_graph_pool_handle())


class XPUGraph(_XPUGraph):
    r"""Wrapper around a XPU graph.

    Arguments:
        keep_graph (bool, optional): If ``keep_graph=False``, the
            executable command graph will be instantiated on GPU at the end of
            ``capture_end`` and the underlying modifiable command graph will be
            destroyed. Note that the executable command graph will not be
            instantiated at the end of ``capture_end`` in this
            case. Instead, it will be instantiated via an explicit called
            to ``instantiate`` or automatically on the first call to
            ``replay`` if ``instantiate`` was not already called. Calling
            ``instantiate`` manually before ``replay`` is recommended to
            prevent increased latency on the first call to ``replay``.

    """

    def __new__(cls, keep_graph: bool = False) -> Self:
        return super().__new__(cls, keep_graph)

    def capture_begin(self, pool: Optional[_POOL_HANDLE] = None) -> None:
        r"""Begin capturing XPU work on the current xpu stream.

        Typically, you shouldn't call ``capture_begin`` yourself.
        Use :class:`~torch.xpu.graph`, which call ``capture_begin`` internally.

        Arguments:
            pool (optional): Token (returned by :func:`~torch.xpu.graph_pool_handle` or
                :meth:`other_Graph_instance.pool()<torch.xpu.XPUGraph.pool>`) that hints this graph may share memory
                with the indicated pool.
        """
        super().capture_begin(pool=pool)

    def capture_end(self) -> None:
        r"""End XPU graph capture on the current stream.

        After ``capture_end``, ``replay`` may be called on this instance.

        Typically, you shouldn't call ``capture_end`` yourself.
        Use :class:`~torch.xpu.graph`, which call ``capture_end`` internally.
        """
        super().capture_end()

    def instantiate(self) -> None:
        r"""Instantiate the XPU graph. Will be called by
        ``capture_end`` if ``keep_graph=False``, or by ``replay`` if
        ``keep_graph=True`` and ``instantiate`` has not already been
        explicitly called. Does not destroy the xpu modify command graph returned
        by ``raw_xpu_graph``.
        """
        super().instantiate()

    def replay(self) -> None:
        r"""Replay the XPU work captured by this graph."""
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
        r"""Enable debugging mode for XPUGraph.debug_dump."""
        return super().enable_debug_mode()

    def debug_dump(self, debug_path: str) -> None:
        r"""
        Arguments:
            debug_path (required): Path to dump the graph to.

        Calls a debugging function to dump the graph if the debugging is
        enabled via XPUGraph.enable_debug_mode()
        """
        return super().debug_dump(debug_path)

    def raw_xpu_graph(self) -> int:
        r"""Returns the underlying xpuGraph_t. ``keep_graph`` must be True.

        XPU doesn't provide APIs to manipulate this object.
        """  # noqa: B950
        return super().raw_xpu_graph()

    def raw_xpu_graph_exec(self) -> int:
        r"""Returns the underlying xpuGraphExec_t. ``instantiate`` must have been called if ``keep_graph`` is True, or ``capture_end`` must have been called if ``keep_graph`` is False. If you call ``instantiate()`` after ``raw_xpu_graph_exec()``, the previously returned xpuGraphExec_t will be destroyed. It is your responsibility not to use this object after destruction.

        XPU doesn't provide APIs to manipulate this object.
        """  # noqa: B950
        return super().raw_xpu_graph_exec()


class graph:
    r"""Context-manager that captures XPU work into a :class:`torch.xpu.XPUGraph` object for later replay.

    Arguments:
        xpu_graph (torch.xpu.XPUGraph): Graph object used for capture.
        pool (optional): Opaque token (returned by a call to :func:`~torch.xpu.graph_pool_handle()` or
            :meth:`other_Graph_instance.pool()<torch.xpu.XPUGraph.pool>`) hinting this graph's capture
            may share memory from the specified pool.
        stream (torch.xpu.Stream, optional): If supplied, will be set as the current stream in the context.
            If not supplied, ``graph`` sets its own internal side stream as the current stream in the context.

    .. note::
        For effective memory sharing, if you pass a ``pool`` used by a previous capture and the previous capture
        used an explicit ``stream`` argument, you should pass the same ``stream`` argument to this capture.

    """  # noqa: B950

    default_capture_stream: Optional[torch.xpu.Stream] = None

    def __init__(
        self,
        xpu_graph: XPUGraph,
        pool: Optional[_POOL_HANDLE] = None,
        stream: Optional[torch.xpu.Stream] = None,
    ):
        # Lazy-init of default_capture_stream helps avoid circular-import errors.
        # Not thread safe, but graphs already have the general (explicitly documented)
        # restriction that only one capture may be underway at a time in the process.
        if self.__class__.default_capture_stream is None:
            self.__class__.default_capture_stream = torch.xpu.Stream()

        self.pool: Union[tuple[()], tuple[_POOL_HANDLE]] = (
            () if pool is None else (pool,)
        )
        self.capture_stream = (
            stream if stream is not None else self.__class__.default_capture_stream
        )
        if self.capture_stream is None:
            raise AssertionError("capture_stream must not be None")
        self.stream_ctx = self.capture_stream
        self.xpu_graph = xpu_graph

    def __enter__(self) -> None:
        # Free as much memory as we can for the graph
        torch.xpu.synchronize()

        torch.xpu.empty_cache()
        self.stream_ctx.__enter__()

        self.xpu_graph.capture_begin(*self.pool)

    def __exit__(self, *args: object) -> None:
        self.xpu_graph.capture_end()
        self.stream_ctx.__exit__(*args)


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
    forward XPU work as a XPU graph inside a single autograd node.

    The graphed callable's forward pass also appends
    a backward node to the autograd graph. During backward, this node runs the
    callable's backward work as a XPU graph.

    Therefore, each graphed callable should be a drop-in replacement for its source callable
    in an autograd-enabled training loop.

    See :ref:`Partial-network capture<partial-network-capture>` for detailed use and constraints.

    If you pass a tuple of several callables, their captures will use the same memory pool.

    Arguments:
        callables (torch.nn.Module or Python function, or tuple of these): Callable or callables to graph.
            If you pass a tuple of callables, their order in the tuple must be the same order they'll run
            in the live workload.
        sample_args (tuple of Tensors, or tuple of tuples of Tensors): Samples args for each callable.
            If a single callable was passed, ``sample_args`` must be a single tuple of argument Tensors.
            If a tuple of callables was passed, ``sample_args`` must be tuple of tuples of argument Tensors.
        num_warmup_iters (int): The number of warmup iterations. Currently, ``DataDistributedParallel`` needs
            11 iterations for warm up. Default: ``3``.
        allow_unused_input (bool): If False, specifying inputs that were not used when computing outputs
            (and therefore their grad is always zero) is an error. Defaults to False.
        pool (optional): Token (returned by :func:`~torch.xpu.graph_pool_handle` or
            :meth:`other_Graph_instance.pool()<torch.xpu.XPUGraph.pool>`) that hints this graph may share memory
            with the indicated pool.
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
        :class:`torch.nn.Module`\s passed to :func:`~torch.xpu.make_graphed_callables` must not have module hooks
        registered on them at the time they are passed. However, registering hooks on modules *after* passing them
        through :func:`~torch.xpu.make_graphed_callables` is allowed.

    .. warning::
        When running a graphed callable, you must pass its arguments in the same order and format
        they appeared in that callable's ``sample_args``.

    .. warning::
        The automatic mixed precision is supported in :func:`~torch.xpu.make_graphed_callables` only with disabled
        caching. The context manager `torch.amp.autocast()` must have `cache_enabled=False`.
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
            if not (
                len(c._backward_hooks) == 0
                and len(c._forward_hooks) == 0
                and len(c._forward_pre_hooks) == 0
            ):
                raise RuntimeError(
                    "Modules must not have hooks registered at the time they are passed. However, registering hooks "
                    + "on modules after passing them through make_graphed_callables is allowed."
                )
            if not all(b.requires_grad is False for b in c.buffers()):
                raise RuntimeError(
                    "In any :class:`~torch.nn.Module` passed to "
                    + ":func:`~make_graphed_callables`, only parameters may be trainable. All buffers must have "
                    + "``requires_grad=False``."
                )
        flatten_arg = torch.utils._pytree.arg_tree_leaves(*args)
        flatten_sample_args.append(tuple(flatten_arg))
        if not all(isinstance(arg, torch.Tensor) for arg in flatten_arg):
            raise TypeError(
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

    fwd_graphs = [torch.xpu.XPUGraph() for _ in range(len(callables))]
    bwd_graphs = [torch.xpu.XPUGraph() for _ in range(len(callables))]

    mempool = graph_pool_handle() if pool is None else pool

    # Warmup
    torch.xpu.synchronize()
    with torch.xpu.stream(torch.xpu.Stream()):
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

    torch.xpu.synchronize()

    # Capture forward graphs
    per_callable_static_outputs = []
    per_callable_output_unflatten_spec = []
    for func, args, fwd_graph in zip(callables, _sample_args, fwd_graphs):
        # each graph uses the same mempool
        with torch.xpu.graph(fwd_graph, pool=mempool):
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
        static_grad_outputs = tuple(
            torch.empty_like(o) if o.requires_grad else None for o in static_outputs
        )

        outputs_grad = tuple(o for o in static_outputs if o.requires_grad)
        grad_inputs = None
        if len(outputs_grad) > 0:
            with torch.xpu.graph(bwd_graph, pool=mempool):
                grad_inputs = torch.autograd.grad(
                    outputs=outputs_grad,
                    inputs=tuple(i for i in static_input_surface if i.requires_grad),
                    grad_outputs=tuple(o for o in static_grad_outputs if o is not None),
                    only_inputs=True,
                    allow_unused=allow_unused_input,
                )

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

    def make_graphed_autograd_function(
        fwd_graph: XPUGraph,
        bwd_graph: XPUGraph,
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
                if not isinstance(static_outputs, tuple):
                    raise RuntimeError("static_outputs must be a tuple")
                return tuple(o.detach() for o in static_outputs)

            @staticmethod
            @torch.autograd.function.once_differentiable
            # pyrefly: ignore [bad-override]
            def backward(ctx: object, *grads: Tensor) -> tuple[Tensor, ...]:
                if len(grads) != len(static_grad_outputs):
                    raise RuntimeError(
                        f"Expected {len(static_grad_outputs)} gradients but got {len(grads)}"
                    )
                for g, grad in zip(static_grad_outputs, grads):
                    if g is not None:
                        if g.data_ptr() != grad.data_ptr():
                            g.copy_(grad)
                bwd_graph.replay()

                if not isinstance(static_grad_inputs, tuple):
                    raise RuntimeError("static_grad_inputs must be a tuple")
                return tuple(
                    b.detach() if b is not None else b for b in static_grad_inputs
                )

        def functionalized(*user_args: object) -> object:
            # Runs the new autograd function which replays the XPU graphs
            flatten_user_args = torch.utils._pytree.arg_tree_leaves(*user_args)
            out = Graphed.apply(*(tuple(flatten_user_args) + module_params))
            return torch.utils._pytree.tree_unflatten(out, output_unflatten_spec)

        return functionalized

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
