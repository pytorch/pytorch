import torch
from typing import Callable, Any
import contextlib
from torch.utils._python_dispatch import TorchDispatchMode
from typing import Dict, Tuple, Optional
import weakref
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

_cloned: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_view_funcs: Dict[Tuple[int, int], Callable] = dict()
_sid_to_weakref: Dict[Tuple[int, int], weakref.ReferenceType] = dict()
_ctx_id = 0
_inside_ctx = False

def _get_tid(t) -> Tuple[int, int]:
    return (id(t), t._version)

def _get_sid(t) -> Tuple[int, int]:
    if t._is_view():
        base = t._base
        sid = (base.data_ptr(), base._version)
    else:
        sid = (t.data_ptr(), t._version)
    return sid

class _Handle():
    pass

# Lifetimes

# We want to the lifetime of cloned vars to match the lifetime of the graph
# So everytime we use a saved variable, we store a owning ref to it on the graph

class _swap_with_cloned(saved_tensors_hooks):
    def __init__(self):
        def pack_hook(t):
            tid = _get_tid(t)
            sid = _get_sid(t)

            # _sid_to_weakref tracks which storages+version are saved for backward
            handle: Optional[_Handle] = None
            if sid not in _sid_to_weakref or _sid_to_weakref[sid]() is None:
                handle = _Handle()
                _sid_to_weakref[sid] = weakref.ref(handle)
            else:
                handle = _sid_to_weakref[sid]()

            # We save the original and the clone!
            # We clone if in-place is done, can we free the original though?
            # If during unpack time, there is a entry in _cloned that means
            # it was modified in-place.
            return _ctx_id, t, sid, tid, t._is_view(), handle

        def unpack_hook(tup):
            ctx_id, t, sid, tid, is_view, handle = tup

            assert ctx_id == _ctx_id, (
                "Trying to backward outside of the same context that the graph was created in")

            if handle in _cloned:
                res = _cloned[handle]
                if is_view:
                    view_func = _view_funcs[tid]
                    res = view_func(res)
            else:
                res = t

            return res

        super().__init__(pack_hook, unpack_hook)

def _maybe_clone_arg(t):
    sid = _get_sid(t)
    if sid not in _sid_to_weakref or _sid_to_weakref[sid]() is None:
        return
    handle = _sid_to_weakref[sid]()
    if t._is_view():
        if sid not in _cloned:
            _cloned[handle] = t._base.clone()
    else:
        if sid not in _cloned:
            _cloned[handle] = t.clone()

# TODO: When we replay a view functions, do the modes matter?
# TODO: Find a better way to get these (we may also be missing some).
VIEW_FUNCTIONS = {
    "numpy_T": "self",
    "alias": "self",
    "as_strided": "self",
    "diagonal": "self",
    "expand": "self",
    "permute": "self",
    "select": "self",
    "slice": "self",
    "split": "self",
    "split_with_sizes": "self",
    "squeeze": "self",
    "t": "self",
    "transpose": "self",
    "unfold": "self",
    "unsqueeze": "self",
    "flatten": "self",
    "view": "self",
    "unbind": "self",
    "_indices": "self",
    "_values": "self",
    "indices": "self",
    "values": "self",
    "crow_indices": "self",
    "col_indices": "self",
    "ccol_indices": "self",
    "row_indices": "self",
    # sparse_coo ctor output should really be views of both indices and values,
    # but we only supports making as view of a single variable, and indices is
    # discrete anyways.
    # FIXME: clone indices on construction.
    "sparse_coo_tensor_with_dims_and_tensors": "values",
    "_reshape_alias": "self",
}

class _CloneArgBeforeMutateMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        # We may want to handle out= later
        if func.__name__.split('.')[0][-1] == "_":
            _maybe_clone_arg(args[0])

        rs = func(*args, **kwargs)

        # Eagerly save all the views, so we can replay them if necessary
        if func.__name__.split('.')[0] in VIEW_FUNCTIONS.keys():
            # What about multi-views like chunk?
            tid = _get_tid(rs)
            _view_funcs[tid] = func
        return rs

@contextlib.contextmanager
def allow_mutation_on_saved_tensors():
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
    global _inside_ctx, _ctx_id
    with _swap_with_cloned(), _CloneArgBeforeMutateMode():
        try:
            _ctx_id += 1
            assert not _inside_ctx, "allow_mutation_on_saved_tensors cannot be nested"
            _inside_ctx = True
            yield
        finally:
            _cloned.clear()
            _view_funcs.clear()

            _ctx_id += 1
            _inside_ctx = False
