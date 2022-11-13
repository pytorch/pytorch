import torch
import contextlib
from typing import Callable, Any, Dict, Tuple, Optional, Sequence, List
from torch.utils.hooks import RemovableHandle

__all__ = ["saved_tensors_hooks", "save_on_cpu"]

__all__ = [
    "saved_tensors_hooks",
    "save_on_cpu",
    "disable_saved_tensors_hooks",
    "register_multi_grad_hook",
]

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
