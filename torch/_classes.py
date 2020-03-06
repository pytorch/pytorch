import types
import torch._C

class _Classes(types.ModuleType):
    def __init__(self):
        super(_Classes, self).__init__('torch.classes')

    def __getattr__(self, attr):
        proxy = torch._C._get_custom_class_python_wrapper(attr)
        if proxy is None:
            raise RuntimeError('Class {} not registered!'.format(attr))
        return proxy

    @property
    def loaded_libraries(self):
        return torch.ops.loaded_libraries

    def load_library(self, path):
        """
        Loads a shared library from the given path into the current process.

        The library being loaded may run global initialization code to register
        custom classes with the PyTorch JIT runtime. This allows dynamically
        loading custom classes. For this, you should compile your class
        and the static registration code into a shared library object, and then
        call ``torch.classes.load_library('path/to/libcustom.so')`` to load the
        shared object.

        After the library is loaded, it is added to the
        ``torch.classes.loaded_libraries`` attribute, a set that may be inspected
        for the paths of all libraries loaded using this function.

        Arguments:
            path (str): A path to a shared library to load.
        """
        torch.ops.load_library(path)

# The classes "namespace"
classes = _Classes()
