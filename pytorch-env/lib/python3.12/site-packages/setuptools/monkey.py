"""
Monkey patching of distutils.
"""

from __future__ import annotations

import functools
import inspect
import platform
import sys
import types
from importlib import import_module
from typing import TypeVar

import distutils.filelist


_T = TypeVar("_T")

__all__: list[str] = []
"""
Everything is private. Contact the project team
if you think you need this functionality.
"""


def _get_mro(cls):
    """
    Returns the bases classes for cls sorted by the MRO.

    Works around an issue on Jython where inspect.getmro will not return all
    base classes if multiple classes share the same name. Instead, this
    function will return a tuple containing the class itself, and the contents
    of cls.__bases__. See https://github.com/pypa/setuptools/issues/1024.
    """
    if platform.python_implementation() == "Jython":
        return (cls,) + cls.__bases__
    return inspect.getmro(cls)


def get_unpatched(item: _T) -> _T:
    lookup = (
        get_unpatched_class
        if isinstance(item, type)
        else get_unpatched_function
        if isinstance(item, types.FunctionType)
        else lambda item: None
    )
    return lookup(item)


def get_unpatched_class(cls):
    """Protect against re-patching the distutils if reloaded

    Also ensures that no other distutils extension monkeypatched the distutils
    first.
    """
    external_bases = (
        cls for cls in _get_mro(cls) if not cls.__module__.startswith('setuptools')
    )
    base = next(external_bases)
    if not base.__module__.startswith('distutils'):
        msg = "distutils has already been patched by %r" % cls
        raise AssertionError(msg)
    return base


def patch_all():
    import setuptools

    # we can't patch distutils.cmd, alas
    distutils.core.Command = setuptools.Command

    _patch_distribution_metadata()

    # Install Distribution throughout the distutils
    for module in distutils.dist, distutils.core, distutils.cmd:
        module.Distribution = setuptools.dist.Distribution

    # Install the patched Extension
    distutils.core.Extension = setuptools.extension.Extension
    distutils.extension.Extension = setuptools.extension.Extension
    if 'distutils.command.build_ext' in sys.modules:
        sys.modules[
            'distutils.command.build_ext'
        ].Extension = setuptools.extension.Extension

    patch_for_msvc_specialized_compiler()


def _patch_distribution_metadata():
    from . import _core_metadata

    """Patch write_pkg_file and read_pkg_file for higher metadata standards"""
    for attr in (
        'write_pkg_info',
        'write_pkg_file',
        'read_pkg_file',
        'get_metadata_version',
        'get_fullname',
    ):
        new_val = getattr(_core_metadata, attr)
        setattr(distutils.dist.DistributionMetadata, attr, new_val)


def patch_func(replacement, target_mod, func_name):
    """
    Patch func_name in target_mod with replacement

    Important - original must be resolved by name to avoid
    patching an already patched function.
    """
    original = getattr(target_mod, func_name)

    # set the 'unpatched' attribute on the replacement to
    # point to the original.
    vars(replacement).setdefault('unpatched', original)

    # replace the function in the original module
    setattr(target_mod, func_name, replacement)


def get_unpatched_function(candidate):
    return candidate.unpatched


def patch_for_msvc_specialized_compiler():
    """
    Patch functions in distutils to use standalone Microsoft Visual C++
    compilers.
    """
    from . import msvc

    if platform.system() != 'Windows':
        # Compilers only available on Microsoft Windows
        return

    def patch_params(mod_name, func_name):
        """
        Prepare the parameters for patch_func to patch indicated function.
        """
        repl_prefix = 'msvc14_'
        repl_name = repl_prefix + func_name.lstrip('_')
        repl = getattr(msvc, repl_name)
        mod = import_module(mod_name)
        if not hasattr(mod, func_name):
            raise ImportError(func_name)
        return repl, mod, func_name

    # Python 3.5+
    msvc14 = functools.partial(patch_params, 'distutils._msvccompiler')

    try:
        # Patch distutils._msvccompiler._get_vc_env
        patch_func(*msvc14('_get_vc_env'))
    except ImportError:
        pass
