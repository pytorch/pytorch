import inspect
from typing import Generic, TypeVar

from torch.distributed.rpc import PyRRef


T = TypeVar("T")
RRefTypeVar = Generic[T]


try:
    # Combine the implementation class and the type class.
    class RRef(PyRRef, RRefTypeVar):
        pass
except TypeError as exc:
    # TypeError: metaclass conflict: the metaclass of a derived class
    # must be a (non-strict) subclass of the metaclasses of all its bases
    class RRefMeta(PyRRef.__class__, RRefTypeVar.__class__):
        pass

    # Combine the implementation class and the type class.
    class RRef(PyRRef, RRefTypeVar, metaclass=RRefMeta):
        pass


def is_rref(ann):
    if ann is RRef:
        raise RuntimeError(
            "Attempted to use RRef without a "
            "contained type. Please add a contained type, e.g. "
            "RRef[int]"
        )
    return getattr(ann, "__origin__", None) is RRef

# Install docstrings from `PyRRef` to `RRef`.
#
# This is for the fact that pybind11 generates the parameter
# `self` as type `rpc.PyRRef`, so a `:inherited-members:`
# under `.. autoclass:: RRef` does not work.
# we have to do the following process to replacee `rpc.PyRRef` with `rpc.RRef`.
#
def method_factory(method_name, docstring):
    def method(self, *args, **kwargs):
        return getattr(super(RRef, self), method_name)(*args, **kwargs)

    method.__doc__ = docstring
    return method


for method_name, method in inspect.getmembers(PyRRef):
    # Ignore magic methods, except "__str__".
    if method_name.startswith("_") and method_name != "__str__":
        continue

    # Get pybind11 generated docstring.
    # It's like,
    """
    to_here(self: torch.distributed.rpc.PyRRef, timeout: float=-1.0) -> object

        Blocking call that copies the value of the RRef from the owner
        to the local node and returns it. If the current node is the
        owner, returns a reference to the local value.
    """
    docstring = getattr(method, "__doc__", None)
    assert docstring is not None, "RRef user-facing methods should all have docstrings."

    # Do surgery on pybind11 generated docstrings.
    docstring = docstring.replace("torch.distributed.rpc.PyRRef", "torch.distributed.rpc.RRef")

    # Attach user-facing RRef method with modified docstring.
    new_method = method_factory(method_name, docstring)
    setattr(RRef, method_name, new_method)
