# mypy: allow-untyped-defs
import contextlib
import weakref

import torch
from torch.types import Storage
from torch._utils import _StateDictStager, staging_copy_context

from ._pin_memory_utils import pin_shared_mem, unpin_memory


@contextlib.contextmanager
def staging_context(stager, non_blocking: bool = False):
    # Assert that only one stager can be active at a time
    assert staging_copy_context.stager is None, "Another stager is already active. Only one stager can be active at a time."
    staging_copy_context.stager = stager
    staging_copy_context.non_blocking = non_blocking
    yield
    staging_copy_context.stager = None

class StateDictStager(_StateDictStager):
    """
    A class for optimizing storage objects during staging for async checkpointing.

    StateDictStager intercepts the deep copy for staging and applies optimizations 
    like memory sharing and pinning to improve performance. It caches storage objects 
    to avoid redundant copies and can be configured to automatically share memory 
    (for multi-process usage) and pin memory (for faster CPU-GPU transfers).

    This class extends torch._utils._StateDictStager and is designed to be used
    with the staging_context context manager.

    Attributes:
        pin_memory (bool): Whether to pin CPU memory for faster CPU-GPU transfers
        share_memory (bool): Whether to share memory across processes
        _cached_storage_mapping (dict[int, Storage]): Maps storage IDs to optimized CPU storages
    """
   
    def __init__(self, pin_memory: bool = False, share_memory: bool = False):
        self.pin_memory = pin_memory
        self.share_memory = share_memory
        self._cached_storage_mapping: dict[int, Storage] = {}  # Mapping from original storage IDs to CPU storages
        
    def _cleanup_storage(self, storage_id):
        if storage_id in self._cached_storage_mapping:
            del self._cached_storage_mapping[storage_id]

    def _unpin_cpu_storage(self, cpu_storage_id):
        unpin_memory(cpu_storage_id)


    def stage_storage(self, storage : Storage):
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
        storage_id = storage.data_ptr()

        # Check if we've already cached this storage
        if storage_id in self._cached_storage_mapping:
            cached_storage = self._cached_storage_mapping[storage_id]
            if cached_storage.size() == storage.size():
                # Reuse cached storage but update with new data
                cached_storage.copy_(storage, non_blocking=staging_copy_context.non_blocking)
                return cached_storage

        # Create new CPU storage
        new_storage = type(storage)(storage.nbytes(), device="cpu").copy_(storage, non_blocking=staging_copy_context.non_blocking)

        # Apply memory optimizations
        if self.share_memory:
            new_storage.share_memory_()
        
        if self.pin_memory:
            assert torch.cuda.is_available(), "pin_memory requires CUDA"
            pin_shared_mem(new_storage.data_ptr(), new_storage.nbytes())
            # Set up a weak reference to unpin when cpu storage is garbage collected
            weakref.finalize(new_storage, self._unpin_cpu_storage, new_storage.data_ptr())
        
        # Cache the storage
        self._cached_storage_mapping[storage_id] = new_storage

        # Set up a weak reference to clean the cache mapping when it's garbage collected
        weakref.finalize(storage, self._cleanup_storage, storage_id)
        return new_storage
