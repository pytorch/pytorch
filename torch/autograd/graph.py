import torch
from typing import Callable, Any
import contextlib
from torch.utils._python_dispatch import TorchDispatchMode
from typing import Dict, Tuple, Optional, Set
import weakref

__all__ = ["saved_tensors_hooks", "save_on_cpu", "allow_mutation_on_saved_tensors"]


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

_cloned: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_original: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_tid_to_weakhandle: Dict[Tuple[int, int], weakref.ReferenceType] = dict()
_sid_to_tid: Dict[Tuple[int, int], Set[Tuple[int, int]]] = dict()
_ctx_id = 0
_inside_ctx = False

def _get_tid(t) -> Tuple[int, int]:
    return (id(t), t.data_ptr(), t._version)

def _get_sid(t) -> Tuple[int, int]:
    return (t.data_ptr(), t._version)

class _Handle():
    pass

class _swap_with_cloned(saved_tensors_hooks):
    def __init__(self):
        def pack_hook(t):
            tid = _get_tid(t)
            sid = _get_sid(t)
            # Tensors saved for backward have an entry in _tid_to_weakhandle
            handle: Optional[_Handle] = None

            # Save aliasing information
            if sid not in _sid_to_tid:
                _sid_to_tid[sid] = set()
            _sid_to_tid[sid].add(tid)

            # NB: The same tensor (of the same version) can be saved multiple times
            if tid not in _tid_to_weakhandle or _tid_to_weakhandle[tid]() is None:
                handle = _Handle()
                _tid_to_weakhandle[tid] = weakref.ref(handle)
                _original[handle] = t
            else:
                # Store an additional handle
                handle = _tid_to_weakhandle[tid]()
            return _ctx_id, handle

        def unpack_hook(tup):
            ctx_id, handle = tup
            assert ctx_id == _ctx_id, (
                "Trying to backward outside of the 'allow_mutation_on_saved_tensors' context"
                "in which the graph was originally recorded.")
            if handle in _cloned:
                res = _cloned[handle]
            else:
                res = _original[handle]
            return res

        super().__init__(pack_hook, unpack_hook)

class _CloneArgBeforeMutateMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        # Bug? Maybe something to do with wrapped number + dispatcher, not sure why yet
        # We don't really need this unless we try to print though, but keeping for now
        if type(args[0]) is int:
            args = [torch.tensor(args[0]), *args[1:]]

        # (only for in-place ops now, we may want to handle out= later)
        if func.__name__.split('.')[0][-1] == "_":
            # The first argument is assumed to be modified in-place
            tid = _get_tid(args[0])
            sid = _get_sid(args[0])
            if sid in _sid_to_tid:
                for tid in _sid_to_tid[sid]:
                    if tid not in _tid_to_weakhandle:
                        # It's never been saved
                        continue
                    handle = _tid_to_weakhandle[tid]()
                    if handle is None or handle in _cloned:
                        # It's been saved, but backward was run OR
                        # The same exactly tensor has been cloned already
                        continue
                    _cloned[handle] = _original[handle].clone()
                    del _original[handle]
            else:
                # this can happen with math views, I'm not sure why yet
                assert not args[0]._is_view()

        rs = func(*args, **kwargs)
        return rs

@contextlib.contextmanager
def allow_mutation_on_saved_tensors():
    """Context manager under which mutating tensors saved for backward is allowed

    Under this context manager, tensors saved for backward are cloned on mutation,
    so the original version can still be used during backward. Normally, mutating a tensor
    saved for backward will result in an error raised when it's used during backward.
    """
    global _inside_ctx, _ctx_id
    with _swap_with_cloned(), _CloneArgBeforeMutateMode():
        try:
            _ctx_id += 1
            if _inside_ctx:
                raise RuntimeError("allow_mutation_on_saved_tensors contexts cannot be nested")
            _inside_ctx = True
            yield
        finally:
            _cloned.clear()
            _ctx_id += 1
            _inside_ctx = False
