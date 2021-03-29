## @package lazy_dyndep
# Module caffe2.python.lazy_dyndep





import os
from caffe2.python import dyndep, lazy


def RegisterOpsLibrary(name):
    """Registers a dynamic library that contains custom operators into Caffe2.

    Since Caffe2 uses static variable registration, you can optionally load a
    separate .so file that contains custom operators and registers that into
    the caffe2 core binary. In C++, this is usually done by either declaring
    dependency during compilation time, or via dynload. This allows us to do
    registration similarly on the Python side.

    Unlike dyndep.InitOpsLibrary, this does not actually parse the c++ file
    and refresh operators until caffe2 is called in a fashion which requires
    operators. In some large codebases this saves a large amount of time
    during import.

    It is safe to use within a program that also uses dyndep.InitOpsLibrary

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
    global _LAZY_IMPORTED_DYNDEPS
    _LAZY_IMPORTED_DYNDEPS.add(name)


_LAZY_IMPORTED_DYNDEPS = set()
_error_handler = None


def SetErrorHandler(handler):
    """Registers an error handler for errors from registering operators

    Since the lazy registration may happen at a much later time, having a dedicated
    error handler allows for custom error handling logic. It is highly
    recomended to set this to prevent errors from bubbling up in weird parts of the
    code.

    Args:
        handler: a function that takes an exception as a single handler.
    Returns:
        None
    """

    global _error_handler
    _error_handler = handler


def GetImportedOpsLibraries():
    _import_lazy()
    return dyndep.GetImportedOpsLibraries()


def _import_lazy():
    global _LAZY_IMPORTED_DYNDEPS
    if not _LAZY_IMPORTED_DYNDEPS:
        return
    for name in list(_LAZY_IMPORTED_DYNDEPS):
        try:
            dyndep.InitOpLibrary(name, trigger_lazy=False)
        except BaseException as e:
            if _error_handler:
                _error_handler(e)
        finally:
            _LAZY_IMPORTED_DYNDEPS.remove(name)

lazy.RegisterLazyImport(_import_lazy)
