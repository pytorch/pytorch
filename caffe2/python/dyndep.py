## @package dyndep
# Module caffe2.python.dyndep





import ctypes
import os
from threading import Lock
from caffe2.python import core, extension_loader


def InitOpsLibrary(name, trigger_lazy=True):
    """Loads a dynamic library that contains custom operators into Caffe2.

    Since Caffe2 uses static variable registration, you can optionally load a
    separate .so file that contains custom operators and registers that into
    the caffe2 core binary. In C++, this is usually done by either declaring
    dependency during compilation time, or via dynload. This allows us to do
    registration similarly on the Python side.

    Args:
        name: a name that ends in .so, such as "my_custom_op.so". Otherwise,
            the command will simply be ignored.
    Returns:
        None
    """
    if not os.path.exists(name):
        # Note(jiayq): if the name does not exist, instead of immediately
        # failing we will simply print a warning, deferring failure to the
        # time when an actual call is made.
        print('Ignoring {} as it is not a valid file.'.format(name))
        return
    _init_impl(name, trigger_lazy=trigger_lazy)


_IMPORTED_DYNDEPS = set()
dll_lock = Lock()


def GetImportedOpsLibraries():
    return _IMPORTED_DYNDEPS


def _init_impl(path, trigger_lazy=True):
    with dll_lock:
        _IMPORTED_DYNDEPS.add(path)
        with extension_loader.DlopenGuard():
            ctypes.CDLL(path)
        # reinitialize available ops
        core.RefreshRegisteredOperators(trigger_lazy)
