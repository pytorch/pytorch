import ctypes
import types
import torch._C

from torch._ops import dl_open_guard

class _Classes(types.ModuleType):
    def __init__(self):
        super(_Classes, self).__init__('torch.classes')
        self.loaded_libraries = set()

    def __getattr__(self, attr):
        proxy = torch._C._get_custom_class_python_wrapper(attr)
        if proxy is None:
            raise RuntimeError('Class {} not registered!'.format(attr))
        return proxy

    def load_library(self, path):
        """
        Loads a shared library from the given path into the current process.

        The library being loaded may run global initialization code to register
        custom operators with the PyTorch JIT runtime. This allows dynamically
        loading custom operators. For this, you should compile your operator
        and the static registration code into a shared library object, and then
        call ``torch.ops.load_library('path/to/libcustom.so')`` to load the
        shared object.

        After the library is loaded, it is added to the
        ``torch.ops.loaded_libraries`` attribute, a set that may be inspected
        for the paths of all libraries loaded using this function.

        Arguments:
            path (str): A path to a shared library to load.
        """
        path = torch._utils_internal.resolve_library_path(path)
        with dl_open_guard():
            # Import the shared library into the process, thus running its
            # static (global) initialization code in order to register custom
            # operators with the JIT.
            ctypes.CDLL(path)
        self.loaded_libraries.add(path)

# The classes "namespace"
classes = _Classes()
