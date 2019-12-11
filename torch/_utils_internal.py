from __future__ import absolute_import, division, print_function, unicode_literals

import os
import inspect
import warnings

# this arbitrary-looking assortment of functionality is provided here
# to have a central place for overrideable behavior. The motivating
# use is the FB build environment, where this source file is replaced
# by an equivalent.

if os.path.basename(os.path.dirname(__file__)) == 'shared':
    torch_parent = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
else:
    torch_parent = os.path.dirname(os.path.dirname(__file__))


def get_file_path(*path_components):
    return os.path.join(torch_parent, *path_components)


def get_file_path_2(*path_components):
    return os.path.join(*path_components)


def get_writable_path(path):
    return path


def prepare_multiprocessing_environment(path):
    pass


def resolve_library_path(path):
    return os.path.realpath(path)


def get_source_lines_and_file(obj):
    """
    Wrapper around inspect.getsourcelines and inspect.getsourcefile.

    Returns: (sourcelines, file_lino, filename)
    """
    filename = None  # in case getsourcefile throws
    try:
        filename = inspect.getsourcefile(obj)
        sourcelines, file_lineno = inspect.getsourcelines(obj)
    except OSError as e:
        raise OSError((
            "Can't get source for {}. TorchScript requires source access in order to carry out compilation. " +
            "Make sure original .py files are available. Original error: {}").format(filename, e))

    return sourcelines, file_lineno, filename


def check_module_version_greater_or_equal(module, req_version_tuple, error_if_malformed=True):
    '''
    Check if a module's version satisfies requirements

    Usually, a module's version string will be like 'x.y.z', which would be represented
    as a tuple (x, y, z), but sometimes it could be an unexpected format. If the version
    string does not match the given tuple's format up to the length of the tuple, then
    error and exit or emit a warning.

    Args:
        module: the module to check the version of
        req_version_tuple: tuple (usually of ints) representing the required version
        error_if_malformed: whether we should exit if module version string is malformed

    Returns:
        requirement_is_met: bool
    '''
    try:
        version_strs = module.__version__.split('.')
        # Cast module version fields to match the types of the required version
        module_version = tuple(
            type(req_field)(version_strs[idx]) for idx, req_field in enumerate(req_version_tuple)
        )
        requirement_is_met = module_version >= req_version_tuple

    except:
        message = ("'%s' module version string is malformed '%s' and cannot be compared"
            " with tuple %s" % (module.__name__, module.__version__, str(req_version_tuple)))
        if error_if_malformed:
            raise Exception(message)
        else:
            warnings.warn(message + ', but continuing assuming that requirement is met')
            requirement_is_met = True

    return requirement_is_met


TEST_MASTER_ADDR = '127.0.0.1'
TEST_MASTER_PORT = 29500
