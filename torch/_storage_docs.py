"""Adds docstrings to Storage functions"""

import torch._C
from torch._C import _add_docstr as add_docstr


storage_classes = [
    'ByteStorageBase',
]


def add_docstr_all(method, docstr):
    for cls_name in storage_classes:
        cls = getattr(torch._C, cls_name)
        try:
            add_docstr(getattr(cls, method), docstr)
        except AttributeError:
            pass


add_docstr_all('from_file',
               """
from_file(filename, shared=False, size=0) -> Storage

If `shared` is `True`, then memory is shared between all processes.
All changes are written to the file. If `shared` is `False`, then the changes on
the storage do not affect the file.

`size` is the number of elements in the storage. If `shared` is `False`,
then the file must contain at least `size * sizeof(Type)` bytes
(`Type` is the type of storage). If `shared` is `True` the file will be
created if needed.

Args:
    filename (str): file name to map
    shared (bool): whether to share memory
    size (int): number of elements in the storage
""")
