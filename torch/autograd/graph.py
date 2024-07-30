import abc
import contextlib
import functools
import logging
import threading
from collections import defaultdict, deque
from typing import (
    Any,
    Callable,
    cast,
    Deque,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Literal,
    MutableMapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    TYPE_CHECKING,
    Union,
)
from typing_extensions import TypeAlias
from weakref import WeakKeyDictionary, WeakValueDictionary

import torch
from torch.autograd.variable import Variable
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.hooks import RemovableHandle


if TYPE_CHECKING:
    from torch._ops import OpOverload


__all__ = [
    "saved_tensors_hooks",
    "save_on_cpu",
    "disable_saved_tensors_hooks",
    "register_multi_grad_hook",
    "allow_mutation_on_saved_tensors",
    "Node",
    "GradientEdge",
    "get_gradient_edge",
    "increment_version",
]


log = logging.getLogger(__name__)


class Node(abc.ABC):
    @abc.abstractmethod
    def name(self) -> str:
        r"""Return the name.

        Example::

            >>> import torch
            >>> a = torch.tensor([0., 0., 0.], requires_grad=True)
            >>> b = a.clone()
            >>> assert isinstance(b.grad_fn, torch.autograd.graph.Node)
            >>> print(b.grad_fn.name())
            CloneBackward0
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def next_functions(self) -> Tuple[Tuple[Optional["Node"], int], ...]:
        raise NotImplementedError

    @abc.abstractmethod
    def metadata(self) -> dict:
        r"""Return the metadata."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def _input_metadata(self) -> List[Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def _register_hook_dict(self, tensor: torch.Tensor) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def register_hook(self, fn: Callable[..., Any]) -> RemovableHandle:
        r"""Register a backward hook.

        The hook will be called every time a gradient with respect to the
        Node is computed. The hook should have the following signature::

            hook(grad_inputs: Tuple[Tensor], grad_outputs: Tuple[Tensor]) -> Tuple[Tensor] or None


        The hook should not modify its argument, but it can optionally return
        a new gradient which will be used in place of :attr:`grad_inputs`.

        This function returns a handle with a method ``handle.remove()``
        that removes the hook from the module.

        .. note::
            See :ref:`backward-hooks-execution` for more information on how when this hook
            is executed, and how its execution is ordered relative to other hooks.

        Example::

            >>> import torch
            >>> a = torch.tensor([0., 0., 0.], requires_grad=True)
            >>> b = a.clone()
            >>> assert isinstance(b.grad_fn, torch.autograd.graph.Node)
            >>> handle = b.grad_fn.register_hook(lambda gI, gO: (gO[0] * 2,))
            >>> b.sum().backward(retain_graph=True)
            >>> print(a.grad)
            tensor([2., 2., 2.])
            >>> handle.remove() # Removes the hook
            >>> a.grad = None
            >>> b.sum().backward(retain_graph=True)
            >>> print(a.grad)
            tensor([1., 1., 1.])
        """
        raise NotImplementedError

    @abc.abstractmethod
    def register_prehook(self, fn: Callable[..., Any]) -> RemovableHandle:
        r"""Register a backward pre-hook.

        The hook will be called every time a gradient with respect to the
        Node is computed. The hook should have the following signature::

            hook(grad_outputs: Tuple[Tensor]) -> Tuple[Tensor] or None

        The hook should not modify its argument, but it can optionally return
        a new gradient which will be used in place of :attr:`grad_outputs`.

        This function returns a handle with a method ``handle.remove()``
        that removes the hook from the module.

        .. note::
            See :ref:`backward-hooks-execution` for more information on how when this hook
            is executed, and how its execution is ordered relative to other hooks.

        Example::

            >>> a = torch.tensor([0., 0., 0.], requires_grad=True)
            >>> b = a.clone()
            >>> assert isinstance(b.grad_fn, torch.autograd.graph.Node)
            >>> handle = b.grad_fn.register_prehook(lambda gI: (gI[0] * 2,))
            >>> b.sum().backward(retain_graph=True)
            >>> print(a.grad)
            tensor([2., 2., 2.])
            >>> handle.remove()
            >>> a.grad = None
            >>> b.sum().backward(retain_graph=True)
            >>> print(a.grad)
            tensor([1., 1., 1.])
        """
        raise NotImplementedError

    @classmethod
    def __subclasshook__(cls, subclass: type) -> bool:
        if cls is Node and (
            (
                subclass is not None
                and subclass is getattr(torch._C._functions, subclass.__name__, None)
            )
            or issubclass(subclass, torch.autograd.function.BackwardCFunction)
        ):
            return True
        return NotImplemented


def _get_grad_fn_or_grad_acc(t: Union[torch.Tensor, "GradientEdge"]) -> Node:
    if isinstance(t, GradientEdge):
        return t.node
    if t.requires_grad and t.grad_fn is None:
        node = t.view_as(t).grad_fn.next_functions[0][0]  # type: ignore[union-attr]
    else:
        node = t.grad_fn
    assert node is not None
    return node


class GradientEdge(NamedTuple):
    """Object representing a given gradient edge within the autograd graph.

    To get the gradient edge where a given Tensor gradient will be computed,
    you can do ``edge = autograd.graph.get_gradient_edge(tensor)``.
    """

    node: Node
    output_nr: int


def get_gradient_edge(tensor: torch.Tensor) -> GradientEdge:
    """Get the gradient edge for computing the gradient of the given Tensor.

    In particular, it is equivalent to call
    ``g = autograd.grad(loss, input)`` and ``g = autograd.grad(loss, get_gradient_edge(input))``.
    """
    if not tensor.requires_grad:
        raise RuntimeError(
            "It is not possible to get the gradient edge for a Tensor "
            "that does not require gradients",
        )
    grad_fn = _get_grad_fn_or_grad_acc(tensor)

    # Note that output_nr default to 0 which is the right value
    # for the AccumulateGrad node.
    return GradientEdge(grad_fn, tensor.output_nr)


def increment_version(tensor: torch.Tensor) -> None:
    """Update autograd metadata tracking whether the given Tensor was modified in place.

    This is to enable more accurate error checking within the autograd engine.
    It is already done automatically by PyTorch functions and within custom Function
    when mark_dirty() is called appropriately so you only need to call this explicitly
    if you are doing inplace operation on the Tensor data in a way that Pytorch doesn't
    know about. For example a custom kernel that reads the Tensor data_ptr and modifies
    the memory inplace based on this pointer.

    Note that incrementing the version counter multiple times for a single inplace operation
    is not problematic.
    """
    torch._C._increment_version(tensor)


class saved_tensors_hooks:
    """Context-manager that sets a pair of pack / unpack hooks for saved tensors.

    Use this context-manager to define how intermediary results of an operation
    should be packed before saving, and unpacked on retrieval.

    In that context, the ``pack_hook`` function will be called everytime an
    operation saves a tensor for backward (this includes intermediary results
    saved using
    :func:`~torch.autograd.function._ContextMethodMixin.save_for_backward` but
    also those recorded by a PyTorch-defined operation). The output of
    ``pack_hook`` is then stored in the computation graph instead of the
    original tensor.

    The ``unpack_hook`` is called when the saved tensor needs to be accessed,
    namely when executing :func:`torch.Tensor.backward()` or
    :func:`torch.autograd.grad()`. It takes as argument the *packed* object
    returned by ``pack_hook`` and should return a tensor which has the same
    content as the original tensor (passed as input to the corresponding
    ``pack_hook``).

    The hooks should have the following signatures:

        pack_hook(tensor: Tensor) -> Any

        unpack_hook(Any) -> Tensor

    where the return value of ``pack_hook`` is a valid input to ``unpack_hook``.

    In general, you want ``unpack_hook(pack_hook(t))`` to be equal to ``t`` in terms
    of value, size, dtype and device.

    Example::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
        >>> def pack_hook(x):
        ...     print("Packing", x)
        ...     return x
        >>>
        >>> def unpack_hook(x):
        ...     print("Unpacking", x)
        ...     return x
        >>>
        >>> a = torch.ones(5, requires_grad=True)
        >>> b = torch.ones(5, requires_grad=True) * 2
        >>> with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        ...     y = a * b
        Packing tensor([1., 1., 1., 1., 1.], requires_grad=True)
        Packing tensor([2., 2., 2., 2., 2.], grad_fn=<MulBackward0>)
        >>> y.sum().backward()
        Unpacking tensor([1., 1., 1., 1., 1.], requires_grad=True)
        Unpacking tensor([2., 2., 2., 2., 2.], grad_fn=<MulBackward0>)

    .. warning ::
        Performing an inplace operation on the input to either hooks may lead
        to undefined behavior.

    .. warning ::
        Only one pair of hooks is allowed at a time. When recursively nesting this
        context-manager, only the inner-most pair of hooks will be applied.
    """

    def __init__(
        self,
        pack_hook: Callable[[torch.Tensor], Any],
        unpack_hook: Callable[[Any], torch.Tensor],
    ) -> None:
        self.pack_hook = pack_hook
        self.unpack_hook = unpack_hook

    def __enter__(self) -> None:
        torch._C._autograd._push_saved_tensors_default_hooks(
            self.pack_hook, self.unpack_hook
        )

    def __exit__(self, *args: object) -> None:
        torch._C._autograd._pop_saved_tensors_default_hooks()


class save_on_cpu(saved_tensors_hooks):
    """Context manager under which tensors saved by the forward pass will be stored on cpu, then retrieved for backward.

    When performing operations within this context manager, intermediary
    results saved in the graph during the forward pass will be moved to CPU,
    then copied back to the original device when needed for the backward pass.
    If the graph was already on CPU, no tensor copy is performed.

    Use this context-manager to trade compute for GPU memory usage (e.g.
    when your model doesn't fit in GPU memory during training).

    Args:
        pin_memory (bool): If ``True`` tensors will be saved to CPU pinned memory
                           during packing and copied to GPU asynchronously during unpacking.
                           Defaults to ``False``.
                           Also see :ref:`cuda-memory-pinning`.


    Example::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
        >>> a = torch.randn(5, requires_grad=True, device="cuda")
        >>> b = torch.randn(5, requires_grad=True, device="cuda")
        >>> c = torch.randn(5, requires_grad=True, device="cuda")
        >>>
        >>> def f(a, b, c):
        ...     prod_1 = a * b           # a and b are saved on GPU
        ...     with torch.autograd.graph.save_on_cpu():
        ...         prod_2 = prod_1 * c  # prod_1 and c are saved on CPU
        ...     y = prod_2 * a           # prod_2 and a are saved on GPU
        ...     return y
        >>>
        >>> y = f(a, b, c)
        >>> del a, b, c  # for illustration only
        >>> # the content of a, b, and prod_2 are still alive on GPU
        >>> # the content of prod_1 and c only live on CPU
        >>> y.sum().backward()  # all CPU tensors are moved back to GPU, for backward
        >>> # all intermediary tensors are released (deleted) after the call to backward
    """

    def __init__(self, pin_memory: bool = False, device_type: str = "cuda") -> None:
        device_module = getattr(torch, device_type, torch.cuda)

        def pack_to_cpu(tensor: torch.Tensor) -> Tuple[torch.device, torch.Tensor]:
            if not pin_memory:
                return (tensor.device, tensor.cpu())
            packed = torch.empty(
                tensor.size(),
                dtype=tensor.dtype,
                layout=tensor.layout,
                pin_memory=(device_module.is_available() and not tensor.is_sparse),
            )
            packed.copy_(tensor)
            return (tensor.device, packed)

        def unpack_from_cpu(packed: Tuple[torch.device, torch.Tensor]) -> torch.Tensor:
            device, tensor = packed
            return tensor.to(device, non_blocking=pin_memory)

        super().__init__(pack_to_cpu, unpack_from_cpu)


@contextlib.contextmanager
def disable_saved_tensors_hooks(error_message: str) -> Generator[None, None, None]:
    """Context-manager that disables the saved tensors default hooks feature.

    Useful for if you are creating a feature that does not work with saved
    tensors default hooks.

    Args:
        error_message (str): When saved tensors default hooks are used when they
                             have been are disabled, a RuntimeError with this
                             error message gets raised.

    Example::

        >>> # xdoctest: +SKIP(failing)
        >>> message = "saved tensors default hooks are disabled"
        >>> with torch.autograd.graph.disable_saved_tensors_hooks(message):
        ...     # Raises RuntimeError: saved tensors default hooks are disabled
        ...     with torch.autograd.graph.save_on_cpu():
        ...         pass
    """
    maybe_prev_message = None
    try:
        maybe_prev_message = (
            torch._C._autograd._saved_tensors_hooks_get_disabled_error_message()
        )
        torch._C._autograd._saved_tensors_hooks_disable(error_message)
        yield
    finally:
        # See NOTE: [disabled_error_message invariant]
        if maybe_prev_message is None:
            torch._C._autograd._saved_tensors_hooks_enable()
        else:
            torch._C._autograd._saved_tensors_hooks_disable(maybe_prev_message)


class _MultiHandle(RemovableHandle):
    handles: Tuple[RemovableHandle, ...]

    def __init__(self, handles: Tuple[RemovableHandle, ...]) -> None:
        self.handles = handles

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()

    def __getstate__(self) -> Tuple[RemovableHandle, ...]:
        return self.handles

    def __setstate__(self, state: Tuple[RemovableHandle, ...]) -> None:
        self.handles = state


def register_multi_grad_hook(
    tensors: Sequence[torch.Tensor],
    fn: Union[
        Callable[[Sequence[Optional[torch.Tensor]]], None],
        Callable[[torch.Tensor], None],
    ],
    *,
    mode: Literal["all", "any"] = "all",
) -> RemovableHandle:
    r"""Register a multi-grad backward hook.

    There are two supported modes: ``"all"`` and ``"any"``.

    Under the ``"all"`` mode, the hook will be called after gradients with respect to every tensor in
    :attr:`tensors` have been computed. If a tensor is in :attr:`tensors` but
    is not part of the graph, or if a tensor is not needed to compute the gradients
    for any ``inputs`` specified for the current ``.backward()`` or ``.grad()`` call,
    this tensor will be ignored and the hook will not wait for its gradient to be
    computed.

    After every non-ignored tensor's gradient has been computed, :attr:`fn` will be
    called with those gradients. ``None`` will be passed for tensors that did not
    have their gradients computed.

    Under the ``"any"`` mode, the hook will be called after the first gradient
    with respect to a tensor in :attr:`tensors` has been computed. The hook
    will be called with that gradient as its argument.

    The hook should not modify its arguments.

    This function returns a handle with a method ``handle.remove()`` that removes the hook.

    .. note::
        See :ref:`backward-hooks-execution` for more information on how when this hook
        is executed, and how its execution is ordered relative to other hooks.

    Example::

        >>> import torch
        >>>
        >>> a = torch.rand(2, 3, requires_grad=True)
        >>> b = torch.rand(2, 3, requires_grad=True)
        >>> c = a * b
        >>> d = a * b
        >>>
        >>> def fn(grads):
        ...     print([g is not None for g in grads])
        ...
        >>> torch.autograd.graph.register_multi_grad_hook((a, b, c, d), fn)
        >>>
        >>> c.sum().backward(retain_graph=True)
        [True, True, True, False]
        >>> c.sum().backward(inputs=(a,), retain_graph=True)
        [True, False, True, False]
        >>>
    """
    supported_modes = ("all", "any")
    lock = threading.Lock()

    if mode not in supported_modes:
        raise ValueError(f"Expects mode to be one of {supported_modes} but got {mode}")

    if mode == "all":
        count: Dict[int, int] = {}
        nb_calls = None
        buffer: Dict[int, List[Optional[torch.Tensor]]] = {}

        grad_fns = list(map(_get_grad_fn_or_grad_acc, tensors))
        len_tensors = len(tensors)

        def get_inner_hook(idx: int) -> Callable[[torch.Tensor], None]:
            def inner_hook(grad: torch.Tensor) -> None:
                nonlocal count, nb_calls, buffer, fn
                id = torch._C._current_graph_task_id()
                assert (
                    id != -1
                ), "expected this hook to be called inside a backward call"
                count[id] = count.get(id, 0)
                buffer[id] = buffer.get(id, [None] * len_tensors)

                with lock:
                    curr_count, count[id] = count[id], count[id] + 1

                    if curr_count == 0:
                        # On the first call, compute the actual nb_calls and buffer
                        nb_calls = sum(
                            map(torch._C._will_engine_execute_node, grad_fns)
                        )

                buffer[id][idx] = grad

                assert nb_calls is not None
                if curr_count == nb_calls - 1:
                    fn = cast(Callable[[Sequence[Optional[torch.Tensor]]], None], fn)
                    fn(buffer[id])
                    del count[id]
                    del buffer[id]

            return inner_hook

        handles = tuple(
            t.register_hook(get_inner_hook(i)) for i, t in enumerate(tensors)
        )
    elif mode == "any":
        fn = cast(Callable[[torch.Tensor], None], fn)
        ran_hook: Dict[int, bool] = defaultdict(bool)

        @functools.wraps(fn)
        def wrapped_fn(grad: torch.Tensor) -> None:
            nonlocal ran_hook
            id = torch._C._current_graph_task_id()
            assert id != -1, "expected this hook to be called inside a backward call"
            with lock:
                prev, ran_hook[id] = ran_hook[id], True
            if prev:
                return
            fn(grad)

        handles = tuple(
            tensor.register_hook(wrapped_fn)
            for tensor in tensors
            if tensor.requires_grad
        )

    return _MultiHandle(handles)  # type: ignore[possibly-undefined]


# NOTE [Allow mutation on tensors saved for backward]
#
# 1. Tensor gets saved for backward
#    - remember the python object id and the version of the tensor
#    - remember aliasing information (data_ptr of base + version)
#    - save the original so we control its lifetime
# 2. Any time a tensor gets in-placed
#    - for each tensor aliased to it:
#      - check using its object id and version to see if it has been saved
#      - if it has been saved, clone it
#      - delete the reference to the original
# 3. during backward
#    - if the clone exists, the tensor must've been modified in-place
_allow_mutation_on_saved_tensors_enabled: bool = False


_TID: TypeAlias = Tuple[int, int, int]
_SID: TypeAlias = Tuple[int, int]


def _get_tid(tensor: torch.Tensor) -> _TID:
    # FIXME: This is almost definitely a bug.
    if isinstance(
        tensor,
        (
            torch._subclasses.fake_tensor.FakeTensor,
            torch._subclasses.functional_tensor.FunctionalTensor,
        ),
    ):
        data_ptr = 0
    else:
        data_ptr = tensor.data_ptr()
    return (id(tensor), data_ptr, tensor._version)


def _get_sid(tensor: torch.Tensor) -> _SID:
    # FIXME: This is almost definitely a bug.
    if isinstance(
        tensor,
        (
            torch._subclasses.fake_tensor.FakeTensor,
            torch._subclasses.functional_tensor.FunctionalTensor,
        ),
    ):
        data_ptr = 0
    else:
        data_ptr = tensor.data_ptr()
    return (data_ptr, tensor._version)


class _Handle:
    pass


class _swap_with_cloned(saved_tensors_hooks):
    def __init__(self, ctx: "_AllowMutationOnSavedContext") -> None:
        def pack_hook(tensor: torch.Tensor) -> _Handle:
            tid = _get_tid(tensor)
            sid = _get_sid(tensor)
            # Tensors saved for backward have an entry in _tid_to_weakhandle
            handle: Optional[_Handle] = None

            # Save aliasing information
            ctx.sid_to_tid[sid].add(tid)

            # NB: The same tensor (of the same version) can be saved multiple times
            if tid not in ctx.tid_to_weakhandle:
                handle = _Handle()
                ctx.tid_to_weakhandle[tid] = handle
                ctx.original[handle] = tensor
            else:
                # Store an additional strong reference to the handle
                handle = ctx.tid_to_weakhandle[tid]
            return handle

        def unpack_hook(handle: _Handle) -> torch.Tensor:
            error_msg = (
                "Trying to backward outside of the 'allow_mutation_on_saved_tensors' context"
                "in which the graph was originally recorded."
            )
            assert _allow_mutation_on_saved_tensors_enabled, error_msg
            if handle in ctx.cloned:
                res = ctx.cloned[handle]
            else:
                assert handle in ctx.original, error_msg
                res = ctx.original[handle]
            return res

        super().__init__(pack_hook, unpack_hook)


class _CloneArgBeforeMutateMode(TorchDispatchMode):
    def __init__(self, ctx: "_AllowMutationOnSavedContext") -> None:
        self.ctx = ctx

    def __torch_dispatch__(
        self,
        func: "OpOverload",
        types: Iterable[type],
        args: Tuple[Any, ...] = (),
        kwargs: Optional[Dict[Any, Any]] = None,
    ) -> Any:
        kwargs = kwargs or {}

        for idx, arg in enumerate(func._schema.arguments):
            if arg.alias_info is not None and arg.alias_info.is_write:
                t = kwargs["out"] if arg.is_out else args[idx]
                tid = _get_tid(t)
                sid = _get_sid(t)
                ctx = self.ctx
                if sid in ctx.sid_to_tid:
                    for tid in ctx.sid_to_tid[sid]:
                        if tid not in ctx.tid_to_weakhandle:
                            # We know that if tid is in sid_to_tid, then it must also be in
                            # tid_to_weakhandle. However, it is possible for the tensor to be
                            # saved at one point, but cleared by backward before it is modified
                            # in-place. Consider the following example:
                            #
                            # >>> a = torch.randn(2, 3, requires_grad=True).clone()
                            # >>> out = (a**2).sum()
                            # >>> out.backward()
                            # >>> a.sin_()
                            continue
                        handle = ctx.tid_to_weakhandle[tid]
                        if handle in ctx.cloned:
                            # The same exact tensor has been cloned already
                            continue
                        ctx.cloned[handle] = ctx.original[handle].clone()
                        del ctx.original[handle]

        return func(*args, **kwargs)


class _AllowMutationOnSavedContext:
    def __init__(self) -> None:
        self.cloned: MutableMapping[_Handle, torch.Tensor] = WeakKeyDictionary()
        self.original: MutableMapping[_Handle, torch.Tensor] = WeakKeyDictionary()
        self.tid_to_weakhandle: MutableMapping[_TID, _Handle] = WeakValueDictionary()
        self.sid_to_tid: Dict[_SID, Set[_TID]] = defaultdict(set)

    def clear(self) -> None:
        self.cloned.clear()
        self.original.clear()
        self.tid_to_weakhandle.clear()
        self.sid_to_tid.clear()


@contextlib.contextmanager
def allow_mutation_on_saved_tensors() -> (
    Generator[_AllowMutationOnSavedContext, None, None]
):
    """Context manager under which mutating tensors saved for backward is allowed.

    Under this context manager, tensors saved for backward are cloned on mutation,
    so the original version can still be used during backward. Normally, mutating a tensor
    saved for backward will result in an error raised when it's used during backward.

    To ensure the correct behavior, both the forward and backward should be run under
    the same context manager.

    Returns:
        An _AllowMutationOnSavedContext object storing the state managed by this
        context manager. This object can be useful for debugging purposes. The state
        managed by the context manager is automatically cleared upon exiting.

    Example::

        >>> import torch
        >>> with torch.autograd.graph.allow_mutation_on_saved_tensors():
        ...     # forward
        ...     a = torch.ones(2, 3, requires_grad=True)
        ...     b = a.clone()
        ...     out = (b**2).sum()
        ...     b.sin_()
        ...     # backward
        ...     out.sum().backward()
        ...
        tensor([[0.8415, 0.8415, 0.8415],
                [0.8415, 0.8415, 0.8415]], grad_fn=<SinBackward0>)
    """
    global _allow_mutation_on_saved_tensors_enabled

    ctx = _AllowMutationOnSavedContext()

    with _swap_with_cloned(ctx), _CloneArgBeforeMutateMode(ctx):
        try:
            if _allow_mutation_on_saved_tensors_enabled:
                raise RuntimeError(
                    "allow_mutation_on_saved_tensors contexts cannot be nested"
                )
            _allow_mutation_on_saved_tensors_enabled = True
            yield ctx
        finally:
            ctx.clear()
            _allow_mutation_on_saved_tensors_enabled = False


def _register_logging_hooks_on_whole_graph(
    t_outputs: Sequence[Union[torch.Tensor, GradientEdge]],
) -> Callable[[], None]:
    grad_fns = list(map(_get_grad_fn_or_grad_acc, t_outputs))

    def iter_graph(roots: List[Node]) -> Iterator[Node]:
        if not roots:
            return
        seen: Set[Node] = set()
        q: Deque[Node] = deque()
        for node in roots:
            if node is not None:
                seen.add(node)
                q.append(node)

        while q:
            node = q.popleft()
            for fn, _ in node.next_functions:
                if fn in seen or fn is None:
                    continue
                seen.add(fn)
                q.append(fn)

            yield node

    def fmt(t: Optional[torch.Tensor]) -> str:
        # Avoid circular import
        from torch.testing._internal.common_utils import dtype_abbrs

        if t is None:
            return "None"
        return f"{dtype_abbrs[t.dtype]}[{', '.join(map(str, t.shape))}]"

    def prehook(grad_outputs: Sequence[Optional[torch.Tensor]]) -> None:
        node = torch._C._current_autograd_node()
        grad_outputs_str = f"[{','.join(fmt(t) for t in grad_outputs)}]"
        log_str = f"Executing: {node} with grad_outputs: {grad_outputs_str}"
        log.debug(log_str)

    handles = []
    for node in iter_graph(grad_fns):
        handles.append(node.register_prehook(prehook))

    def unregister_hooks() -> None:
        for handle in handles:
            handle.remove()

    return unregister_hooks


def _engine_run_backward(
    t_outputs: Sequence[Union[torch.Tensor, GradientEdge]],
    *args: Any,
    **kwargs: Any,
) -> Tuple[torch.Tensor, ...]:
    attach_logging_hooks = log.getEffectiveLevel() <= logging.DEBUG
    if attach_logging_hooks:
        unregister_hooks = _register_logging_hooks_on_whole_graph(t_outputs)
    try:
        return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
            t_outputs, *args, **kwargs
        )  # Calls into the C++ engine to run the backward pass
    finally:
        if attach_logging_hooks:
            unregister_hooks()  # type: ignore[possibly-undefined]
