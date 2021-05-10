
import os
import inspect
import sys
import tempfile

# this arbitrary-looking assortment of functionality is provided here
# to have a central place for overrideable behavior. The motivating
# use is the FB build environment, where this source file is replaced
# by an equivalent.

if sys.executable == 'torch_deploy':
    # __file__ is meaningless in the context of frozen torch used in torch deploy.
    # setting empty torch_parent should allow below functions to operate without crashing,
    # but it's unclear if there is a valid use case for them in the context of deploy.
    torch_parent = ""
else:
    if os.path.basename(os.path.dirname(__file__)) == 'shared':
        torch_parent = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    else:
        torch_parent = os.path.dirname(os.path.dirname(__file__))

def get_file_path(*path_components):
    return os.path.join(torch_parent, *path_components)


def get_file_path_2(*path_components):
    return os.path.join(*path_components)


def get_writable_path(path):
    if os.access(path, os.W_OK):
        return path
    return tempfile.mkdtemp(suffix=os.path.basename(path))



def prepare_multiprocessing_environment(path):
    pass


def resolve_library_path(path):
    return os.path.realpath(path)


def get_source_lines_and_file(obj, error_msg=None):
    """
    Wrapper around inspect.getsourcelines and inspect.getsourcefile.

    Returns: (sourcelines, file_lino, filename)
    """
    filename = None  # in case getsourcefile throws
    try:
        filename = inspect.getsourcefile(obj)
        sourcelines, file_lineno = inspect.getsourcelines(obj)
    except OSError as e:
        msg = (f"Can't get source for {obj}. TorchScript requires source access in "
               "order to carry out compilation, make sure original .py files are "
               "available.")
        if error_msg:
            msg += '\n' + error_msg
        raise OSError(msg) from e

    return sourcelines, file_lineno, filename


TEST_MASTER_ADDR = '127.0.0.1'
TEST_MASTER_PORT = 29500
# USE_GLOBAL_DEPS controls whether __init__.py tries to load
# libtorch_global_deps, see Note [Global dependencies]
USE_GLOBAL_DEPS = True
# USE_RTLD_GLOBAL_WITH_LIBTORCH controls whether __init__.py tries to load
# _C.so with RTLD_GLOBAL during the call to dlopen.
USE_RTLD_GLOBAL_WITH_LIBTORCH = False
