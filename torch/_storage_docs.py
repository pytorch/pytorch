"""Adds docstrings to Storage functions"""

import torch._C
from torch._C import _add_docstr as add_docstr


storage_classes = [
    'DoubleStorageBase',
    'FloatStorageBase',
    'LongStorageBase',
    'IntStorageBase',
    'ShortStorageBase',
    'CharStorageBase',
    'ByteStorageBase',
]


def add_docstr_all(classes, method, docstr):
    for cls_name in classes:
        cls = getattr(torch._C, cls_name)
        add_docstr(getattr(cls, method),
                   docstr.format(
                       cls_name.replace('Base', ''),
                       cls_name.replace('StorageBase', '')))


add_docstr_all(storage_classes, 'from_file',
               """
from_file(filename, shared=False, size=0) -> {0}

If shared is True then memory is shared between all processes. All changes are
written to the file. If shared is False then the changes on the storage do not
affect the file.

Size is the number of elements in the storage. If shared is False then the file
must contain at least `size * sizeof({1})` bytes. If shared is True the file
will be created if needed.

Args:
    filename (str): file name to map
    shared (bool): whether to share memory
    size (int): number of elements in the storage
""")
