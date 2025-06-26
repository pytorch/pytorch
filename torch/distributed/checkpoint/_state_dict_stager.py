# mypy: allow-untyped-defs
import logging
import types
import weakref
from copyreg import dispatch_table
from logging import getLogger
from typing import Any

import torch
import torch.cuda._pin_memory_utils as pin_memory_utils
from torch.storage import UntypedStorage
from torch.utils.weak import WeakIdKeyDictionary


logger = getLogger()
logger.setLevel(logging.INFO)


class StateDictStager:
    """
    A class for optimizing storage objects during staging for async checkpointing.

    StateDictStager stages the state_dict to CPU DRAM while applying optimizations
    like memory sharing and pinning to improve performance. It caches storage objects
    to avoid redundant copies and can be configured to automatically share memory
    (for multi-process usage) and pin memory (for faster CPU-GPU transfers).

    Attributes:
        pin_memory (bool): Whether to pin CPU memory for faster CPU-GPU transfers
        share_memory (bool): Whether to share memory across processes
        _cached_storage_mapping (WeakIdKeyDictionary): Maps storage objects to optimized CPU storages using weak references
    """

    def __init__(self, pin_memory: bool = False, share_memory: bool = False):
        if pin_memory and not torch.cuda.is_available():
            logger.warning(
                "Ignoring pin_memory flag for checkpoint staging as pinning memory"
                "requires CUDA, but CUDA is not available. "
            )
            self.pin_memory = False
        else:
            self.pin_memory = pin_memory
        self.share_memory = share_memory
        # Mapping from original storage objects to CPU storages using weak references
        self._cached_storage_mapping = WeakIdKeyDictionary()

        def _deepcopy_atomic(x, _):
            return x

        def _deepcopy_list(x, memo):
            y: list = []
            memo[id(x)] = y
            append = y.append
            for a in x:
                append(self.deepcopy_with_tensor_offload(a, memo))
            return y

        def _deepcopy_tuple(x, memo):
            y = [self.deepcopy_with_tensor_offload(a, memo) for a in x]
            # We're not going to put the tuple in the memo, but it's still important we
            # check for it, in case the tuple contains recursive mutable structures.
            try:
                return memo[id(x)]
            except KeyError:
                pass

            # Check if any elements changed during deepcopy
            for k, j in zip(x, y):
                if k is not j:
                    # At least one element changed, create new tuple
                    return tuple(y)

            # No elements changed, return original tuple
            return x

        def _deepcopy_dict(x, memo):
            y: dict = {}
            memo[id(x)] = y
            for key, value in x.items():
                y[self.deepcopy_with_tensor_offload(key, memo)] = (
                    self.deepcopy_with_tensor_offload(value, memo)
                )
            return y

        def _deepcopy_method(x, memo):  # Copy instance methods
            return type(x)(
                x.__func__, self.deepcopy_with_tensor_offload(x.__self__, memo)
            )

        d: dict[Any, Any] = {}
        self._deepcopy_dispatch = d
        d[type(None)] = _deepcopy_atomic
        d[int] = _deepcopy_atomic
        d[float] = _deepcopy_atomic
        d[bool] = _deepcopy_atomic
        d[complex] = _deepcopy_atomic
        d[bytes] = _deepcopy_atomic
        d[str] = _deepcopy_atomic
        d[types.CodeType] = _deepcopy_atomic
        d[type] = _deepcopy_atomic
        d[range] = _deepcopy_atomic
        d[types.BuiltinFunctionType] = _deepcopy_atomic
        d[types.FunctionType] = _deepcopy_atomic
        d[weakref.ref] = _deepcopy_atomic
        d[property] = _deepcopy_atomic
        d[types.MethodType] = _deepcopy_method
        d[dict] = _deepcopy_dict
        d[tuple] = _deepcopy_tuple
        d[list] = _deepcopy_list

    def _stage_untyped_storage(
        self, storage: UntypedStorage, non_blocking: bool = False
    ):
        """
        Called from the hooked storage_deepcopy function in torch.Tensor.__deepcopy__.

        This method handles the storage optimization logic for the StagingStateDict class.
        It checks if the storage has already been cached, and if so, reuses it.
        Otherwise, it creates a new CPU storage and applies memory optimizations.

        Args:
            storage: The storage to optimize

        Returns:
            The optimized storage
        """
        # Check if we've already cached this storage
        if storage in self._cached_storage_mapping:
            cached_storage = self._cached_storage_mapping[storage]
            assert cached_storage.size() == storage.size(), (
                "For async checkpointing,  We cache storages in DRAM and reuse them."
                "Cached storage size does not match original storage size."
                "This should never happen as we track the original storage weakref "
                "and clean up the cache storage. Please report this to PyTorch Distributed Checkpointing."
            )
            # Reuse cached storage but update with new data
            cached_storage.copy_(storage, non_blocking=non_blocking)
            return cached_storage

        # Create new CPU storage
        if self.share_memory:
            new_storage = type(storage)._new_shared(storage.size(), device="cpu")
        else:
            new_storage = type(storage)(storage.size(), device="cpu")

        if self.pin_memory and new_storage.nbytes() > 0:
            pin_memory_utils.pin_memory(new_storage.data_ptr(), new_storage.nbytes())
            # Set up a weak reference to unpin when cpu storage is garbage collected
            f = weakref.finalize(
                new_storage, pin_memory_utils.unpin_memory, new_storage.data_ptr()
            )
            # This makes sure that the finalizer is not called after
            # cuda context is destroyed.
            f.atexit = False

        new_storage.copy_(storage, non_blocking=non_blocking)

        # Cache the storage - WeakIdKeyDictionary will automatically clean up when storage is garbage collected
        self._cached_storage_mapping[storage] = new_storage
        return new_storage

    @torch.no_grad()
    def stage(
        self,
        state_dict: dict[str, Any],
        non_blocking: bool = False,
    ) -> dict[str, Any]:
        return self.deepcopy_with_tensor_offload(state_dict, non_blocking=non_blocking)

    def _offload_tensor(self, x, memo, non_blocking=False):
        """
        Deep copy a PyTorch tensor with optimized storage handling.

        This method creates a CPU copy of a tensor while applying memory optimizations
        like sharing and pinning based on the StateDictStager configuration.

        Args:
            x: The tensor to copy
            memo: Memo dictionary for tracking already copied objects
            non_blocking: Whether to perform non-blocking copies where possible

        Returns:
            A CPU copy of the tensor with optimized storage
        """
        # Create a new empty tensor on CPU
        y = x.new_empty([], device="cpu")

        # Store in memo dict early to handle recursive references
        d = id(x)
        memo[d] = y

        if type(x) is torch.Tensor or x.data_ptr() != 0:
            # Try to get the untyped storage and optimize it
            untyped_storage = x.untyped_storage()
            copied_storage = self._stage_untyped_storage(
                untyped_storage, non_blocking=non_blocking
            )
            # Set the tensor data using the optimized storage
            y.set_(copied_storage, x.storage_offset(), x.size(), x.stride())

        # Copy any attributes the tensor might have
        if hasattr(x, "__dict__"):
            for attr_name, attr_value in x.__dict__.items():
                setattr(
                    y,
                    attr_name,
                    self.deepcopy_with_tensor_offload(
                        attr_value, memo, non_blocking=non_blocking
                    ),
                )

        if hasattr(x, "__slots__"):
            for slot in x.__slots__:
                if hasattr(x, slot):
                    setattr(
                        y,
                        slot,
                        self.deepcopy_with_tensor_offload(
                            getattr(x, slot), memo, non_blocking=non_blocking
                        ),
                    )

        return y

    @torch.no_grad()
    def deepcopy_with_tensor_offload(self, x, memo=None, _nil=[], non_blocking=False):  # noqa: B006
        """Deep copy operation on arbitrary Python objects with special handling for PyTorch tensors.

        This implementation extends the standard deepcopy functionality to handle PyTorch tensors
        and their storages in a way that optimizes memory usage and performance, similar to the
        stage method. It applies memory sharing and pinning optimizations based on the StateDictStager
        configuration.

        Args:
            x: The object to deep copy
            memo: Memo dictionary for tracking already copied objects
            _nil: Sentinel value for memo dictionary
            non_blocking: Whether to perform non-blocking copies where possible

        Returns:
            A deep copy of the input object with optimized tensor storage handling
        """
        if memo is None:
            memo = {}

        d = id(x)
        y = memo.get(d, _nil)
        if y is not _nil:
            return y

        cls = type(x)

        # tensors and subclasses of tensors are handled separately
        if isinstance(x, torch.Tensor):
            y = self._offload_tensor(x, memo, non_blocking=non_blocking)

        # Use the dispatch table for standard types
        copier = self._deepcopy_dispatch.get(cls)
        if copier is not None:
            y = copier(x, memo)
        else:
            if issubclass(cls, type):
                y = self._deepcopy_dispatch[type](x, memo)
            else:
                copier = getattr(x, "__deepcopy__", None)
                if copier is not None:
                    y = copier(memo)
                else:
                    reductor = dispatch_table.get(cls)
                    if reductor:
                        rv = reductor(x)
                    else:
                        reductor = getattr(x, "__reduce_ex__", None)
                        if reductor is not None:
                            rv = reductor(4)
                        else:
                            reductor = getattr(x, "__reduce__", None)
                            if reductor:
                                rv = reductor()
                            else:
                                raise RuntimeError(
                                    f"un(deep)copyable object of type {cls}"
                                )
                    if isinstance(rv, str):
                        y = x
                    else:
                        y = self._reconstruct(x, memo, *rv)

        # If is its own copy, don't memoize.
        if y is not x:
            memo[d] = y
            self._keep_alive(x, memo)  # Make sure x lives at least as long as d
        return y

    def _keep_alive(self, x, memo):
        """Keeps a reference to the object x in the memo.

        Because we remember objects by their id, we have
        to assure that possibly temporary objects are kept
        alive by referencing them.
        We store a reference at the id of the memo, which should
        normally not be used unless someone tries to deepcopy
        the memo itself...
        """
        try:
            memo[id(memo)].append(x)
        except KeyError:
            # aha, this is the first one :-)
            memo[id(memo)] = [x]

    def _reconstruct(
        self, x, memo, func, args, state=None, listiter=None, dictiter=None
    ):
        deep = memo is not None
        if deep and args:
            args = (self.deepcopy_with_tensor_offload(arg, memo) for arg in args)
        y = func(*args)
        if deep:
            memo[id(x)] = y

        if state is not None:
            if deep:
                state = self.deepcopy_with_tensor_offload(state, memo)
            if hasattr(y, "__setstate__"):
                y.__setstate__(state)
            else:
                if isinstance(state, tuple) and len(state) == 2:
                    state, slotstate = state
                else:
                    slotstate = None
                if state is not None:
                    y.__dict__.update(state)
                if slotstate is not None:
                    for key, value in slotstate.items():
                        setattr(y, key, value)

        if listiter is not None:
            if deep:
                for item in listiter:
                    item = self.deepcopy_with_tensor_offload(item, memo)
                    y.append(item)
            else:
                for item in listiter:
                    y.append(item)
        if dictiter is not None:
            if deep:
                for key, value in dictiter:
                    key = self.deepcopy_with_tensor_offload(key, memo)
                    value = self.deepcopy_with_tensor_offload(value, memo)
                    y[key] = value
            else:
                for key, value in dictiter:
                    y[key] = value
        return y
