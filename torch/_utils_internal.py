from __future__ import absolute_import, division, print_function, unicode_literals

import os
import inspect

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


TEST_MASTER_ADDR = '127.0.0.1'
TEST_MASTER_PORT = 29500
