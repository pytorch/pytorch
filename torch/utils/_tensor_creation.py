import torch

from torch._torch_docs import factory_common_args


def from_file(
    filename,
    shared=None,
    size=0,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=False,
):
    r"""

    Creates a CPU tensor with a storage backed by a memory-mapped file.

    If ``shared`` is True, then memory is shared between processes. All changes are written to the file.
    If ``shared`` is False, then changes to the tensor do not affect the file.

    ``size`` is the number of elements in the Tensor. If ``shared`` is ``False``, then the file must contain
    at least :math:`size * sizeof(dtype)` bytes. If ``shared`` is `True` the file will be created if needed.

    See :meth:`share_memory_` for a discussion on how the ``shared`` argument differs from sharing memory.

    .. note::
        Only CPU tensors can be mapped to files.

    .. note::
        For now, tensors with storages backed by a memory-mapped file cannot be copied to pinned memory.


    Args:
        filename (str) - file name to map
        shared (bool) - whether to share memory (whether ``MAP_SHARED`` or ``MAP_PRIVATE`` is passed to the
                        underlying `mmap(2) call <https://man7.org/linux/man-pages/man2/mmap.2.html>`_)
        size (int) - number of elements in the tensor

    Keyword args:
        {generator}
        {dtype}
        {layout}
        {device}
        {pin_memory}

    Example::
        >>> t = torch.randn(2, 5, dtype=torch.float64)
        >>> t.numpy().tofile('storage.pt')
        >>> t_mapped = torch.from_file('storage.pt', shared=False, size=10, dtype=torch.float64)
    """.format(
        **factory_common_args
    )
    t = torch._from_file(
        filename,
        shared,
        size,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
    )
    t.untyped_storage().filename = filename
    return t
