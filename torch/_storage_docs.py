"""Adds docstrings to Storage functions"""

import torch._C
from torch._C import _add_docstr as add_docstr


storage_classes = [
    "StorageBase",
]


def add_docstr_all(method, docstr):
    for cls_name in storage_classes:
        cls = getattr(torch._C, cls_name)
        try:
            add_docstr(getattr(cls, method), docstr)
        except AttributeError:
            pass


add_docstr_all(
    "from_file",
    """
from_file(filename, shared=False, size=0) -> Storage

Creates a CPU storage backed by a memory-mapped file.

If ``shared`` is ``True``, then memory is shared between all processes.
All changes are written to the file. If ``shared`` is ``False``, then the changes on
the storage do not affect the file.

``size`` is the number of elements in the storage. If ``shared`` is ``False``,
then the file must contain at least :math:`size * sizeof(Type)` bytes
(``Type`` is the type of storage, in the case of an ``UnTypedStorage`` the file must contain at
least ``size`` bytes). If ``shared`` is ``True`` the file will be created if needed.

Args:
    filename (str): file name to map
    shared (bool): whether to share memory (whether ``MAP_SHARED`` or ``MAP_PRIVATE`` is passed to the
                    underlying `mmap(2) call <https://man7.org/linux/man-pages/man2/mmap.2.html>`_)
    size (int): number of elements in the storage
    """,
)

add_docstr_all(
    "share_memory_",
    """
share_memory_(self) -> Storage

Moves the storage to shared memory.

This is a no-op for storages already in shared memory and for CUDA
storages, which do not need to be moved for sharing across processes.
Storages in shared memory cannot be resized.

Note that to mitigate issues like `this <https://github.com/pytorch/pytorch/issues/95606>`_
it is thread safe to call this function from multiple threads on the same object.
It is NOT thread safe though to call any other function on self without proper
synchronization. Please see :doc:`/notes/multiprocessing` for more details.

.. note::
    When all references to a storage in shared memory are deleted, the associated shared memory
    object will also be deleted. PyTorch has a special cleanup process to ensure that this happens
    even if the current process exits unexpectedly.

    It is worth noting the difference between :meth:`share_memory_` and :meth:`from_file` with ``shared = True``

    * ``share_memory_`` uses `shm_open(3) <https://man7.org/linux/man-pages/man3/shm_open.3.html>`_ to create a
        POSIX shared memory object while :meth:`from_file` uses
        `open(2) <https://man7.org/linux/man-pages/man2/open.2.html>`_ to open the filename passed by the user.
    * Both use an `mmap(2) call <https://man7.org/linux/man-pages/man2/mmap.2.html>`_ with ``MAP_SHARED``
        to map the file/object into the current virtual address space
    * ``share_memory_`` will call ``shm_unlink(3)`` on the file after mapping it to make sure the shared memory
        object is freed when no object has the file open. ``torch.from_file(shared=True)`` does not unlink the
        file. This file is persistent and will remain until it is deleted by the user.


Returns: self
    """
)
