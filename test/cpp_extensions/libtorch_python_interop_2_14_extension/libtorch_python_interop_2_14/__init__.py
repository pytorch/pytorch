# Ensure libtorch / libtorch_python are loaded before importing the extension
# module, whose symbols resolve against them.
import torch  # noqa: F401

from . import _C


def pyobject_roundtrip(t):
    """PyObject -> stable Tensor -> PyObject; result shares storage with ``t``."""
    return _C.pyobject_roundtrip(t)


def pyobject_to_type(t, py_type):
    """Like ``pyobject_roundtrip`` but forces the result's Python type."""
    return _C.pyobject_to_type(t, py_type)


def pyobject_sum(t):
    """from_pyobject -> stable sum over all elements -> to_pyobject."""
    return _C.pyobject_sum(t)


__all__ = ["pyobject_roundtrip", "pyobject_to_type", "pyobject_sum"]
