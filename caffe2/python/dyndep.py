from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import ctypes
from caffe2.python import core, extension_loader


def InitOpsLibrary(name):
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
    if not name.endswith('.so'):
        # TODO(jiayq): deal with extensions on platforms that do not use .so
        # as extensions.
        print('Ignoring {} as it is not an .so file.'.format(name))
        return
    with extension_loader.DlopenGuard():
        ctypes.CDLL(name)
    # reinitialize available ops
    core.RefreshRegisteredOperators()
