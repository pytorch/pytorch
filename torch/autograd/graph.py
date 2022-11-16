import torch
import contextlib
from typing import Callable, Any, Dict, Tuple, Optional, Sequence, List, Set
from torch.utils.hooks import RemovableHandle
from torch.utils._python_dispatch import TorchDispatchMode
from collections import defaultdict
import weakref

__all__ = [
    "saved_tensors_hooks",
    "save_on_cpu",
    "disable_saved_tensors_hooks",
    "register_multi_grad_hook",
    "allow_mutation_on_saved_tensors",
]


def _get_tid(t) -> Tuple[int, int]:
    # Returns a unique identifier for a particular version of a tensor. This is
    # currently used in saved_tensor_hook context managers such as save_on_cpu()
    # and allow_mutation_on_saved_tensors().
    #
    # We claim that id() and t._version are necessary and sufficient to uniquely
    # identify a version of the tensor:
    # - id corresponds to TensorImpl because we have pyobject persistence.
    #   On the other hand, Keeping track of storage via something like data_ptr
    #   is not sufficient because there can be different views to the same storage.
    # - we don't need t.data_ptr() because the only way it can change from
    #   underneath a TensorImpl is 1) if someone uses .data (we're okay with
    #   silently wrong are produced if .data is used) OR 2) if version
    #   counter also changes, e.g. when we perform an in-place view
    #   We choose to omit this from the identifier as to support tensors that
    #   don't have storage, e.g. sparse tensors. It may be better to have
    #   special handling for sparse tensors, but that would also need to consider
    #   additional things like coalescing.
    #
    #   TODO(soulitzer): check sparse correctness, make sure that if contents of sparse
    #   tensor changes, that is also reflected here.
    return (id(t), t._version)

def _get_sid(t) -> Tuple[int, int]:
    # Returns a tuple that uniquely identifies a tensor's storage
    #
    # NB: two tensors that share the same storage may have different
    #     sid if their storage offset is different
    return (t.data_ptr(), t._version)

class _Handle():
    pass

class saved_tensors_hooks():
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
    def __init__(self, pack_hook: Callable[[torch.Tensor], Any], unpack_hook: Callable[[Any], torch.Tensor]):
        self.pack_hook = pack_hook
        self.unpack_hook = unpack_hook

    def __enter__(self):
        torch._C._autograd._push_saved_tensors_default_hooks(self.pack_hook, self.unpack_hook)

    def __exit__(self, *args: Any):
        torch._C._autograd._pop_saved_tensors_default_hooks()

# This would not be necessary if:
# 1) we can rely on tensor subclass authors to provide a helper for testing if two
#    tensors are the same
# 2) extra layers of wrapping automatically die if we use them at a lower interpreter level
#    not sure all tensor subclasses have this concept though
def _functorch_unwrap_to_level(tensor: torch.Tensor, target_level: int) -> torch.Tensor:
    assert target_level != 0, "level 0 is not supported, you should pass -1 instead"
    current_level = torch._C._functorch.maybe_get_level(tensor)
    for _ in range(max(current_level, 0), max(target_level, 0), -1):
        tensor = torch._C._functorch.get_unwrapped(tensor)
    assert torch._C._functorch.maybe_get_level(tensor) == target_level
    return tensor

class save_on_cpu(saved_tensors_hooks):
    """Context-manager under which tensors saved by the forward pass will be
    stored on cpu, then retrieved for backward.

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
    def __init__(self, pin_memory=False):
        # We use weak references here to makes sure that the only owning reference
        # to any tensors that are offloaded have lifetimes that are linked to that
        # of the graph
        self.inner_copies: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self.tid_to_weakhandle: weakref.WeakValueDictionary = weakref.WeakValueDictionary()

        def pack_to_cpu(tensor):
            # NB: Always unwrap the outer layer, which is guaranteed to be a TensorWrapper
            #     or a vanilla tensor. Unwrapping if the level is -1 is a no-op (I think).
            level1 = torch._C._functorch.maybe_get_level(tensor)
            inner = torch._C._functorch._unwrap_for_grad(tensor, level1)
            # NB: level after unwrapping isn't always level - 1, so query again
            level = torch._C._functorch.maybe_get_level(inner)

            tid = _get_tid(_functorch_unwrap_to_level(inner, -1))

            device = inner.device
            handle = None

            if tid in self.tid_to_weakhandle:
                handle = self.tid_to_weakhandle[tid]
                saved_level = torch._C._functorch.maybe_get_level(self.inner_copies[handle])

                if level <= saved_level:
                    # Unsolved problem with this approach:
                    #
                    # Unnecessary copies are made because we need to save the outermost tensor
                    # but as we unwind the stack, we do not know whether the current level of
                    # unwrapping is the outermost (there could be a vmap layer in the middle that
                    # changes what is saved)
                    return (device, handle, level)

            if not pin_memory:
                inner_copy = inner.cpu()
            else:
                inner_copy = torch.empty(
                    inner.size(),
                    dtype=inner.dtype,
                    layout=inner.layout,
                    pin_memory=(torch.cuda.is_available() and not inner.is_sparse))
                inner_copy.copy_(inner)

            handle = _Handle() if handle is None else handle
            self.inner_copies[handle] = inner_copy
            self.tid_to_weakhandle[tid] = handle

            return (device, handle, level)

        def unpack_from_cpu(packed):
            device, handle, level = packed
            ret = _functorch_unwrap_to_level(self.inner_copies[handle], level)
            assert torch._C._functorch.maybe_get_level(ret) == level
            ret = ret.to(device, non_blocking=pin_memory)
            # Ideally we would be completely working with unwrapped tensors during backward,
            # but for a grad transform the wrapping level is the same as that of forward.
            # I think that should not be the case. (Can we fix this?)
            #
            # If saved tensor logic properly detached, we shouldn't have to unwrap and rewrap
            # here at all. The rewrapping here is implicit due to lifting.

            assert torch._C._functorch.maybe_get_level(ret) > level or level == -1, "lifting should happen"

            # Maybe there is a more straightforward way to check the current interpreter level
            # but we are doing .to anyway, so hopefully that should properly lift/kill wrappers.
            return ret

        super().__init__(pack_to_cpu, unpack_from_cpu)

    def __enter__(self):
        torch._C._autograd._push_saved_tensors_default_hooks(self.pack_hook, self.unpack_hook)
        return self

    def __exit__(self, *args: Any):
        # We cannot clear here because that would be bc-breaking: people sometimes run
        # forward using save_on_cpu, but exit the ctx before running backward.
        # Note that this behavior is inconsistent with that of allow_mutation_on_saved ctx
        # which requires that backward also be run inside the ctx.
        #
        # self.inner_copies.clear()
        # self.tid_to_weakhandle.clear()
        torch._C._autograd._pop_saved_tensors_default_hooks()

@contextlib.contextmanager
def disable_saved_tensors_hooks(error_message):
    """Context-manager that disables the saved tensors default hooks feature.

    Useful for if you are creating a feature that does not work with saved
    tensors default hooks.

    Args:
        error_message (str): When saved tensors default hooks are used when they
                             have been are disabled, a RuntimeError with this
                             error message gets raised.

    Example::

        >>> message = "saved tensors default hooks are disabled"
        >>> with torch.autograd.graph.disable_saved_tensors_hooks(message):
        ...     # Raises RuntimeError: saved tensors default hooks are disabled
        ...     with torch.autograd.graph.save_on_cpu():
        ...         pass

    """
    try:
        maybe_prev_message = torch._C._autograd._saved_tensors_hooks_get_disabled_error_message()
        torch._C._autograd._saved_tensors_hooks_disable(error_message)
        yield
    finally:
        # See NOTE: [disabled_error_message invariant]
        if maybe_prev_message is None:
            torch._C._autograd._saved_tensors_hooks_enable()
        else:
            torch._C._autograd._saved_tensors_hooks_disable(maybe_prev_message)


def register_multi_grad_hook(tensors: Sequence[torch.Tensor], fn: Callable[[Sequence[Optional[torch.Tensor]]], None]):
    r"""Registers a multi-grad backward hook.

    The hook will be called after gradients with respect to every tensor in
    :attr:`tensors` have been computed. If a tensor is in :attr:`tensors` but
    is not part of the graph, or if a tensor is not needed to compute the gradients
    for any ``inputs`` specified for the current ``.backward()`` or ``.grad()`` call,
    this tensor will be ignored and the hook will not wait for its gradient to be
    computed.

    After every non-ignored tensor's gradient has been computed, :attr:`fn` will be
    called with those gradients. ``None`` will be passed for tensors that did not
    have their gradients computed.

    The hook should not modify its arguments.

    This function returns a handle with a method ``handle.remove()`` that removes the hook.

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
    count: Dict[int, int] = dict()
    nb_calls = None
    buffer: Dict[int, List[Optional[torch.Tensor]]] = dict()

    def get_grad_fn(t):
        # or grad accumulator
        if t.requires_grad and t.grad_fn is None:
            return t.clone().grad_fn.next_functions[0][0]
        else:
            return t.grad_fn

    grad_fns = list(map(get_grad_fn, tensors))

    def get_inner_hook(idx):
        def inner_hook(grad: torch.Tensor):
            nonlocal count, nb_calls, buffer
            id = torch._C._current_graph_task_id()
            assert id != -1, "expected this hook to be called inside a backward call"
            count[id] = count.get(id, 0)
            buffer[id] = buffer.get(id, [None] * len(tensors))

            if count[id] == 0:
                # On the first call, compute the actual nb_calls and buffer
                nb_calls = sum(torch._C._will_engine_execute_node(g) for g in grad_fns)  # type: ignore[attr-defined]

            buffer[id][idx] = grad
            count[id] += 1

            if count[id] == nb_calls:
                fn(buffer[id])
                del count[id]
                del buffer[id]
        return inner_hook

    class Handle(RemovableHandle):
        handles: Tuple[RemovableHandle, ...]

        def __init__(self, handles: Tuple[RemovableHandle, ...]):
            self.handles = handles

        def remove(self):
            for handle in self.handles:
                handle.remove()

        def __getstate__(self):
            return self.handles

        def __setstate__(self, state):
            self.handles = state

    handles: List[RemovableHandle] = []
    for i, t in enumerate(tensors):
        handles.append(t.register_hook(get_inner_hook(i)))

    return Handle(tuple(handles))


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
_allow_mutation_on_saved_tensors_enabled = False

class _swap_with_cloned(saved_tensors_hooks):
    def __init__(self, ctx):
        def pack_hook(t):
            tid = _get_tid(t)
            sid = _get_sid(t)
            # Tensors saved for backward have an entry in _tid_to_weakhandle
            handle: Optional[_Handle] = None

            # Save aliasing information
            ctx.sid_to_tid[sid].add(tid)

            # NB: The same tensor (of the same version) can be saved multiple times
            if tid not in ctx.tid_to_weakhandle:
                handle = _Handle()
                ctx.tid_to_weakhandle[tid] = handle
                ctx.original[handle] = t
            else:
                # Store an additional strong reference to the handle
                handle = ctx.tid_to_weakhandle[tid]
            return handle

        def unpack_hook(tup):
            handle = tup
            error_msg = (
                "Trying to backward outside of the 'allow_mutation_on_saved_tensors' context"
                "in which the graph was originally recorded.")
            assert _allow_mutation_on_saved_tensors_enabled, error_msg
            if handle in ctx.cloned:
                res = ctx.cloned[handle]
            else:
                assert handle in ctx.original, error_msg
                res = ctx.original[handle]
            return res

        super().__init__(pack_hook, unpack_hook)

class _CloneArgBeforeMutateMode(TorchDispatchMode):
    def __init__(self, ctx):
        self.ctx = ctx

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
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

        rs = func(*args, **kwargs)
        return rs

class _AllowMutationOnSavedContext():
    def __init__(self):
        self.cloned: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self.original: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self.tid_to_weakhandle: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        self.sid_to_tid: Dict[Tuple[int, int], Set[Tuple[int, int, int]]] = defaultdict(set)

    def clear(self):
        self.cloned.clear()
        self.original.clear()
        self.tid_to_weakhandle.clear()
        self.sid_to_tid.clear()

@contextlib.contextmanager
def allow_mutation_on_saved_tensors():
    """Context manager under which mutating tensors saved for backward is allowed

    Under this context manager, tensors saved for backward are cloned on mutation,
    so the original version can still be used during backward. Normally, mutating a tensor
    saved for backward will result in an error raised when it's used during backward.

    To ensure the correct behavior, both the forward and backward should be run under
    the same context manager.

    returns:
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
                raise RuntimeError("allow_mutation_on_saved_tensors contexts cannot be nested")
            _allow_mutation_on_saved_tensors_enabled = True
            yield ctx
        finally:
            ctx.clear()
            _allow_mutation_on_saved_tensors_enabled = False
