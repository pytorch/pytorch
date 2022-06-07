import torch
from typing import Callable, Any
import contextlib
from torch.utils._python_dispatch import TorchDispatchMode, push_torch_dispatch_mode

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
        Packing tensor([1., 1., 1., 1., 1.])
        Packing tensor([2., 2., 2., 2., 2.])
        >>> y.sum().backward()
        Unpacking tensor([1., 1., 1., 1., 1.])
        Unpacking tensor([2., 2., 2., 2., 2.])

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


_cloned = dict()
_use_counts = dict()
_keep_graph = False
class _swap_with_cloned(saved_tensors_hooks):
    def __init__(self):
        def pack_hook(t):
            uid = (t.data_ptr(), t._version)
            if not _keep_graph:
                _use_counts[uid] = _use_counts.get(uid, 0) + 1
            return uid, t

        def unpack_hook(tup):
            uid, t = tup
            if uid in _cloned:
                res = _cloned[uid]
            else:
                res = t
            if not _keep_graph:
                if uid not in _use_counts:
                    raise RuntimeError("If you are trying to backward through the graph a second time "
                                       "or trying to compute higher-order gradients, please specify "
                                       "`keep_graph=True` when enabling allow_mutation_on_saved_tensors")
                _use_counts[uid] -= 1
                if _use_counts[uid] == 0:
                    if uid in _cloned:
                        del _cloned[uid]
                    del _use_counts[uid]
            return res
        super().__init__(pack_hook, unpack_hook)

class _CloneArgBeforeMutateMode(TorchDispatchMode):
    @staticmethod
    def is_mutating(func):
        # We may want to also handle out= later
        return func.__name__.split('.')[0][-1] == "_"

    @staticmethod
    def maybe_clone_arg(t):
        uid = (t.data_ptr(), t._version)
        if uid not in _cloned:
            _cloned[uid] = t.clone()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if _CloneArgBeforeMutateMode.is_mutating(func):
            _CloneArgBeforeMutateMode.maybe_clone_arg(args[0])
        rs = func(*args, **kwargs)
        return rs

@contextlib.contextmanager
def allow_mutation_on_saved_tensors(keep_graph=False):
    """Context manager under which mutating tensors that will be saved
    for backward is allowed.

    When using this context manager, if tensors that are saved for backward
    are mutated, instead of raising an error, a copy of that tensor before
    mutation is stored to be used for the backward pass.

    Args:
        keep_graph (bool): If ``True``, allows one to backward through the graph
                           multiple times and enables higher-order gradients.
                           Defaults to ``False``.
    """
    with _swap_with_cloned(), push_torch_dispatch_mode(_CloneArgBeforeMutateMode):
        global _keep_graph
        # Do we even care about nesting? Maybe just track nesting level and raise an error?
        prev_keep_graph = _keep_graph
        _keep_graph = keep_graph
        try:
            yield
        finally:
            if _keep_graph:
                _cloned.clear()
                _use_counts.clear()
            _keep_graph = prev_keep_graph
