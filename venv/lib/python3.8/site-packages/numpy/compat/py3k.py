"""
Python 3.X compatibility tools.

While this file was originally intended for Python 2 -> 3 transition,
it is now used to create a compatibility layer between different
minor versions of Python 3.

While the active version of numpy may not support a given version of python, we
allow downstream libraries to continue to use these shims for forward
compatibility with numpy while they transition their code to newer versions of
Python.
"""
__all__ = ['bytes', 'asbytes', 'isfileobj', 'getexception', 'strchar',
           'unicode', 'asunicode', 'asbytes_nested', 'asunicode_nested',
           'asstr', 'open_latin1', 'long', 'basestring', 'sixu',
           'integer_types', 'is_pathlib_path', 'npy_load_module', 'Path',
           'pickle', 'contextlib_nullcontext', 'os_fspath', 'os_PathLike']

import sys
import os
from pathlib import Path, PurePath
import io

import abc
from abc import ABC as abc_ABC

try:
    import pickle5 as pickle
except ImportError:
    import pickle

long = int
integer_types = (int,)
basestring = str
unicode = str
bytes = bytes

def asunicode(s):
    if isinstance(s, bytes):
        return s.decode('latin1')
    return str(s)

def asbytes(s):
    if isinstance(s, bytes):
        return s
    return str(s).encode('latin1')

def asstr(s):
    if isinstance(s, bytes):
        return s.decode('latin1')
    return str(s)

def isfileobj(f):
    return isinstance(f, (io.FileIO, io.BufferedReader, io.BufferedWriter))

def open_latin1(filename, mode='r'):
    return open(filename, mode=mode, encoding='iso-8859-1')

def sixu(s):
    return s

strchar = 'U'

def getexception():
    return sys.exc_info()[1]

def asbytes_nested(x):
    if hasattr(x, '__iter__') and not isinstance(x, (bytes, unicode)):
        return [asbytes_nested(y) for y in x]
    else:
        return asbytes(x)

def asunicode_nested(x):
    if hasattr(x, '__iter__') and not isinstance(x, (bytes, unicode)):
        return [asunicode_nested(y) for y in x]
    else:
        return asunicode(x)

def is_pathlib_path(obj):
    """
    Check whether obj is a pathlib.Path object.

    Prefer using `isinstance(obj, os_PathLike)` instead of this function.
    """
    return Path is not None and isinstance(obj, Path)

# from Python 3.7
class contextlib_nullcontext:
    """Context manager that does no additional processing.

    Used as a stand-in for a normal context manager, when a particular
    block of code is only sometimes used with a normal context manager:

    cm = optional_cm if condition else nullcontext()
    with cm:
        # Perform operation, using optional_cm if condition is True
    """

    def __init__(self, enter_result=None):
        self.enter_result = enter_result

    def __enter__(self):
        return self.enter_result

    def __exit__(self, *excinfo):
        pass


def npy_load_module(name, fn, info=None):
    """
    Load a module.

    .. versionadded:: 1.11.2

    Parameters
    ----------
    name : str
        Full module name.
    fn : str
        Path to module file.
    info : tuple, optional
        Only here for backward compatibility with Python 2.*.

    Returns
    -------
    mod : module

    """
    # Explicitly lazy import this to avoid paying the cost
    # of importing importlib at startup
    from importlib.machinery import SourceFileLoader
    return SourceFileLoader(name, fn).load_module()


# Backport os.fs_path, os.PathLike, and PurePath.__fspath__
if sys.version_info[:2] >= (3, 6):
    os_fspath = os.fspath
    os_PathLike = os.PathLike
else:
    def _PurePath__fspath__(self):
        return str(self)

    class os_PathLike(abc_ABC):
        """Abstract base class for implementing the file system path protocol."""

        @abc.abstractmethod
        def __fspath__(self):
            """Return the file system path representation of the object."""
            raise NotImplementedError

        @classmethod
        def __subclasshook__(cls, subclass):
            if PurePath is not None and issubclass(subclass, PurePath):
                return True
            return hasattr(subclass, '__fspath__')


    def os_fspath(path):
        """Return the path representation of a path-like object.
        If str or bytes is passed in, it is returned unchanged. Otherwise the
        os.PathLike interface is used to get the path representation. If the
        path representation is not str or bytes, TypeError is raised. If the
        provided path is not str, bytes, or os.PathLike, TypeError is raised.
        """
        if isinstance(path, (str, bytes)):
            return path

        # Work from the object's type to match method resolution of other magic
        # methods.
        path_type = type(path)
        try:
            path_repr = path_type.__fspath__(path)
        except AttributeError:
            if hasattr(path_type, '__fspath__'):
                raise
            elif PurePath is not None and issubclass(path_type, PurePath):
                return _PurePath__fspath__(path)
            else:
                raise TypeError("expected str, bytes or os.PathLike object, "
                                "not " + path_type.__name__)
        if isinstance(path_repr, (str, bytes)):
            return path_repr
        else:
            raise TypeError("expected {}.__fspath__() to return str or bytes, "
                            "not {}".format(path_type.__name__,
                                            type(path_repr).__name__))
