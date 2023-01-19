"""
Compatibility Support for Python 2.7 and earlier
"""

import sys
import platform

from setuptools.extern import six


def get_all_headers(message, key):
    """
    Given an HTTPMessage, return all headers matching a given key.
    """
    return message.get_all(key)


if six.PY2:
    def get_all_headers(message, key):  # noqa
        return message.getheaders(key)


linux_py2_ascii = (
    platform.system() == 'Linux' and
    six.PY2
)

rmtree_safe = str if linux_py2_ascii else lambda x: x
"""Workaround for http://bugs.python.org/issue24672"""


try:
    from ._imp import find_module, PY_COMPILED, PY_FROZEN, PY_SOURCE
    from ._imp import get_frozen_object, get_module
except ImportError:
    import imp
    from imp import PY_COMPILED, PY_FROZEN, PY_SOURCE  # noqa

    def find_module(module, paths=None):
        """Just like 'imp.find_module()', but with package support"""
        parts = module.split('.')
        while parts:
            part = parts.pop(0)
            f, path, (suffix, mode, kind) = info = imp.find_module(part, paths)

            if kind == imp.PKG_DIRECTORY:
                parts = parts or ['__init__']
                paths = [path]

            elif parts:
                raise ImportError("Can't find %r in %s" % (parts, module))

        return info

    def get_frozen_object(module, paths):
        return imp.get_frozen_object(module)

    def get_module(module, paths, info):
        imp.load_module(module, *info)
        return sys.modules[module]
