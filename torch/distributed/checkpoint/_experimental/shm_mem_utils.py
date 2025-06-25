# mypy: allow-untyped-defs
import os
from logging import getLogger

from .pin_memory_utils import pin_shared_mem, unpin_memory


logger = getLogger()


class SharedMemoryManager:
    def __init__(self):
        self._shm_seg_map = {}
        self._shm_seg_pinned_map = {}

    def create_buffer(self, name: str, size: int, pin_memory: bool = False):
        import multiprocessing.shared_memory

        shm_segment = multiprocessing.shared_memory.SharedMemory(
            create=True, size=size, name=name
        )
        self._shm_seg_map[name] = shm_segment

        if pin_memory:
            pin_shared_mem(self._get_data_ptr(shm_segment), len(shm_segment.buf))
            self._shm_seg_pinned_map[name] = shm_segment
        else:
            # Allocate the shared memory
            os.posix_fallocate(shm_segment._fd, 0, size)

        return shm_segment.buf

    def _get_data_ptr(self, shm_segment):
        # Get the data pointer from the shared memory segment
        # This is a simplified implementation - in practice you might need
        # to handle this differently based on the platform
        return shm_segment.buf.obj

    def close(self):
        # Unpin memory for pinned segments
        for name, shm_segment in self._shm_seg_pinned_map.items():
            unpin_memory(self._get_data_ptr(shm_segment))

        # Close and unlink all shared memory segments
        for name, shm_segment in self._shm_seg_map.items():
            shm_segment.close()
            shm_segment.unlink()

        # Clear the maps
        self._shm_seg_map.clear()
        self._shm_seg_pinned_map.clear()
