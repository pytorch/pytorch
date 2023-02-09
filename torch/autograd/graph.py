import torch
import contextlib
from typing import Callable, Any, Dict, Tuple, Optional, Sequence, List, Set, DefaultDict, NamedTuple
from torch.utils.hooks import RemovableHandle
from torch.utils._python_dispatch import TorchDispatchMode
from collections import defaultdict
from .function import Function
import weakref
import abc

__all__ = [
    "saved_tensors_hooks",
    "save_on_cpu",
    "disable_saved_tensors_hooks",
    "register_multi_grad_hook",
    "allow_mutation_on_saved_tensors",
    "Node",
]

class Node(abc.ABC):
    @abc.abstractmethod
    def name(self) -> str:
        r"""Returns the name.

        Example::

            >>> import torch
            >>> a = torch.tensor([0., 0., 0.], requires_grad=True)
            >>> b = a.clone()
            >>> assert isinstance(b.grad_fn, torch.autograd.graph.Node)
            >>> print(b.grad_fn.name())
            CloneBackward0
        """
        ...

    @property
    @abc.abstractmethod
    def next_functions(self) -> Tuple[Tuple[Optional['Node'], int], ...]:
        ...

    @abc.abstractmethod
    def metadata(self) -> dict:
        r"""Returns the metadata."""
        ...

    @abc.abstractmethod
    def _register_hook_dict(self, tensor: torch.Tensor) -> None:
        ...

    @abc.abstractmethod
    def register_hook(self, fn: Callable[..., Any]) -> RemovableHandle:
        r"""Registers a backward hook.

        The hook will be called every time a gradient with respect to the
        Node is computed. The hook should have the following signature::

            hook(grad_inputs: Tuple[Tensor], grad_outputs: Tuple[Tensor]) -> Tuple[Tensor] or None


        The hook should not modify its argument, but it can optionally return
        a new gradient which will be used in place of :attr:`grad_outputs`.

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
        ...

    @abc.abstractmethod
    def register_prehook(self, fn: Callable[..., Any]) -> RemovableHandle:
        r"""Registers a backward pre-hook.

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
        ...

    @classmethod
    def __subclasshook__(cls, C):
        if cls is Node:
            if ((C is not None and C is getattr(torch._C._functions, C.__name__, None))
                    or issubclass(C, torch.autograd.function.BackwardCFunction)):
                return True
        return NotImplemented

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
    def __init__(self, pack_hook: Callable[[torch.Tensor], Any], unpack_hook: Callable[[Any], torch.Tensor]):
        self.pack_hook = pack_hook
        self.unpack_hook = unpack_hook

    def __enter__(self):
        torch._C._autograd._push_saved_tensors_default_hooks(self.pack_hook, self.unpack_hook)

    def __exit__(self, *args: Any):
        torch._C._autograd._pop_saved_tensors_default_hooks()


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
    def __init__(self, pin_memory=False):
        def pack_to_cpu(tensor):
            if not pin_memory:
                return (tensor.device, tensor.cpu())

            packed = torch.empty(
                tensor.size(),
                dtype=tensor.dtype,
                layout=tensor.layout,
                pin_memory=(torch.cuda.is_available() and not tensor.is_sparse))
            packed.copy_(tensor)
            return (tensor.device, packed)

        def unpack_from_cpu(packed):
            device, tensor = packed
            return tensor.to(device, non_blocking=pin_memory)

        super().__init__(pack_to_cpu, unpack_from_cpu)

# NOTE: [Checkpoint that supports nesting]
#
# Contents:
# - Semantics
# - Mechanics
#
# Checkpointing Semantics (this should just be in the docs)
# =========================================================
#
# Under a checkpoint context, any variables that would've been saved by
# forward under this context would no longer be saved.
#
# Lifetime of recomputed variables
# --------------------------------
#
# Recomputed variables only stay alive for the lifetime of a particular
# backward.
#
# Backward within checkpoint
# --------------------------
#
# If backward is performed inside a checkpoint context...
#
# with checkpoint():
#    y = x.sin()                # saves x
#    z = y.exp()                # saves z
#    torch.autograd.grad(z, z)  # clears z, only x remains saved
#                               # As we exit, clears x only
#
# Nested checkpointing
# --------------------
#
# There is some specially handling for the nested case: the inputs to
# are treated as saved variables in the parent context.
#
# with checkpoint0():
#   with checkpoint1():          # saves `y` in check0
#     y = f(x)                   # f's saved variables are cleared by check1
#   with checkpoint2():          # saves `z` in check0
#     z = g(y)                   # g's saved variables are cleared by check1
#                                # exiting check0, clears `y` and `z`
#                                # whatever f and g save are hidden#
#
# Early stopping
# --------------
#
# Checkpointing Mechanics
# =======================
#
# 1) Doing backward inside
#
# This explains why someone might unpack something packed during recomputation.
# If someone tries to unpack this pack, that means that they are calling backward
# in a checkpoint. No need to be nested. It is okay if we just save the tensor normally,
# i.e. tie the tensor's lifetime to the recomputation graph because the recomputation graph will
# die as soon as we finish recomputation.
#
# 2) Doing backward inside, but we are also nested
#
# This happens when doing backward within checkpoint. I need to have the ability
# to recompute my top-level checkpoint WITHIN the recomputation stack. Consider the case:
#
# checkpoint
#    checkpoint
#       f
#    backward
#
# During recomputation, the inner checkpoint is no longer nested, but we still want
# to pretend as if it is, i.e., we need to save the inputs to parent, so that the
# indices match up. However, if we do backward, during recomputation, we aren't able
# able to recompute the outer parent to obtain the inputs that way since it isn't truly
# nested. What we do is directly save to the current frame as well.
#
# Why is this okay to do?
#
# 2) Two disjoint backwards
#
# This answers the question of how can handle be None?
#
# We want to save in the recomputation graph even if handle is None, i.e., x has already
# been cleared in the original graph. To see why this is the case,
# consider the case where we do two separate backwards on disjoint parts of the graph,
# we need to rerun that first backward to get to the second one.
#
#
# To be done before landing:
# - Test memory usage on CUDA and try to achieve potential the theoretical O(log(n))
#   memory usage reduction
# - Save and restore rng, autocast state
# - Maybe support non-Tensor inputs *args, **kwargs, should just be an API issue.
# - Finish writing the docs here


# NOTE: [Nested Checkpoint Input Handling]
#
# Checkpoint frames need to store the inputs in order to recompute saved tensors.
# We handle the storing of in two ways depending on whether the checkpoint is
# nested.
#
# (1) Non-nested case
# The simplest case is where there is no nesting, and we store args directly
# on the checkpoint frame. If this frame is nested, maybe_args is None.

# (2) Nested case
# In the nested case however, we need to let the parent manage what we save
# and this includes the inputs of the child checkpoints.
# To try to reuse the usual checkpoint logic to handle these, we rely on a dummy
# autograd Function to save inputs as saved tensors. However, Inputs differ from
# normal saved tensors in at that we don't unpack before using them. This means
# that (1) we cannot rely on the packed handle to identify them and so we need a
# couple extra fields so to distinguish them later. (2) we should not detach them.

class NoopSaveInputs(Function):
    # autograd Function that saves inputs and returns them as-is
    # TODO: We need to wrap this to emulate the requires_gradness of the inputs
    @staticmethod
    def forward(*args):
        return args

    @staticmethod
    def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
        ctx.save_for_backward(*inputs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return grad_outputs

class _CheckpointFrame():
    def __init__(self, wrapped_fn):
        # Stores a wrapped function, where the proper contexts are captured
        self.wrapped_fn = wrapped_fn

        # See NOTE: [Nested Checkpoint Input Handling]
        self.maybe_args: Optional[List[torch.Tensor]] = None
        self.args_idx: Optional[List[int]] = None
        self.child_args_idx : List[int] = []
        self.args_handles: Optional[List[Any]] = None

        self.weak_handles: List[weakref.ref[_Handle]] = []

        self.recomputed: DefaultDict[int, weakref.WeakKeyDictionary[_Handle, torch.Tensor]] = \
            defaultdict(weakref.WeakKeyDictionary)
        self.recomp_counter: DefaultDict[int, int] = defaultdict(lambda: 0)
        self.is_recomputed: DefaultDict[int, bool] = defaultdict(lambda: False)

    def __repr__(self):
        return f"Frame({id(self)})"

    def get_args_from_parent(self, parent_frame, gid):
        assert self.args_idx is not None
        out = []
        for idx in self.args_idx:
            handle = parent_frame.weak_handles[idx]()
            assert handle is not None
            if handle:
                out.append(parent_frame.recomputed[gid][handle])
        return out

    def save_args_to_parent(self, parent_frame, args):
        parent_pre_len = len(parent_frame.weak_handles)
        new_args = NoopSaveInputs.apply(*args)  # TODO: fix require grad-ness
        self.args_handles = parent_frame.weak_handles[parent_pre_len: len(parent_frame.weak_handles)]
        indices = list(range(parent_pre_len, len(parent_frame.weak_handles)))
        parent_frame.child_args_idx.extend(indices)
        self.args_idx = indices
        return new_args


class _CheckpointStack(NamedTuple):
    stack: List[_CheckpointFrame]
    is_recompute: bool

# This stack is synced with the saved_tensor_hook stack.
# When _recomputation_hook is pushed onto the hook stack, we also push a new
# empty _CheckpointStack
_checkpoint_stacks: List[_CheckpointStack] = \
    [_CheckpointStack(stack=[], is_recompute=False)]

class _StopRecomputationError(Exception):
    pass

class _recomputation_hook(saved_tensors_hooks):
    def __init__(self, target_frame_ref: weakref.ref[_CheckpointFrame], gid: int):
        def pack_hook(x):
            target_frame = target_frame_ref()
            assert target_frame is not None
            recomp_idx = target_frame.recomp_counter[gid]
            target_frame.recomp_counter[gid] += 1
            handle = target_frame.weak_handles[recomp_idx]()

            if handle is not None:
                # See Checkpointing Mechanics (3) to see when handle can be None
                target_frame.recomputed[gid][handle] = \
                    x if recomp_idx in target_frame.child_args_idx else x.detach()

            if target_frame.recomp_counter[gid] == len(target_frame.weak_handles):
                raise _StopRecomputationError()
            # See Checkpointing Mechanics (1)
            return x.detach()

        def unpack_hook(x):
            return x

        super().__init__(pack_hook, unpack_hook)

class _checkpoint_hook(saved_tensors_hooks):
    def __init__(self):
        def pack_hook(x):
            # Snapshot the state of the current checkpoint stack
            current_frames, is_recompute = _checkpoint_stacks[-1]
            top_frame = current_frames[-1]
            handle = _Handle()
            top_frame.weak_handles.append(weakref.ref(handle))
            return handle, tuple(current_frames)

        def unpack_hook(saved):
            handle, frames = saved

            top_frame = frames[-1]
            gid = torch._C._current_graph_task_id()
            assert gid != -1, "checkpoint: expected unpack to be called inside a backward call"

            for i in range(len(frames)):
                frame = frames[i]
                if frame.is_recomputed[gid]:
                    continue
                # See NOTE [Nested Checkpoint Input Handling]
                if frame.maybe_args is None:
                    args = frame.get_args_from_parent(frames[i - 1], gid)
                else:
                    args = frame.maybe_args
                try:
                    _checkpoint_stacks.append(_CheckpointStack(stack=[], is_recompute=True))
                    # pass gid in in case we do reentrant backward
                    with _recomputation_hook(weakref.ref(frame), gid), torch.autograd.enable_grad():
                        frame.wrapped_fn(*args)
                except _StopRecomputationError as e:
                    _checkpoint_stacks.pop()
                    pass
                frame.is_recomputed[gid] = True

            return top_frame.recomputed[gid][handle]

        super().__init__(pack_hook, unpack_hook)

def _get_wrapped_fn(fn):
    # Capture the current context, so we can replay it
    def wrapped(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapped

def _checkpoint(fn, *args, **kwargs):
    curr_stack, is_curr_stack_recompute = _checkpoint_stacks[-1]
    new_frame = _CheckpointFrame(_get_wrapped_fn(fn))

    # See NOTE [Nested Checkpoint Input Handling]
    if len(curr_stack) > 0:
        args = new_frame.save_args_to_parent(curr_stack[-1], args)
    elif is_curr_stack_recompute:
        # See Checkpointing Mechanics (2)
        args = NoopSaveInputs.apply(*args)
        new_frame.maybe_args = args  # type: ignore[assignment]
    else:
        new_frame.maybe_args = args  # type: ignore[assignment]

    curr_stack.append(new_frame)
    with _checkpoint_hook():
        ret = fn(*args, **kwargs)
    curr_stack.pop()
    return ret


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

        >>> # xdoctest: +SKIP(failing)
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

def _get_tid(t) -> Tuple[int, int, int]:
    return (id(t), t.data_ptr(), t._version)

def _get_sid(t) -> Tuple[int, int]:
    return (t.data_ptr(), t._version)

class _Handle():
    pass

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
