import inspect
from typing import Dict, List

import torch
from ..exc import unimplemented
from ..utils import istype
from .base import VariableTracker


class DistributedVariable(VariableTracker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not DistributedVariable.is_available():
            unimplemented("torch.distributed package is not available!")

    @staticmethod
    def is_available():
        # check if the distributed package is available or not
        return torch.distributed.is_available()


def is_from_local(value):
    if not DistributedVariable.is_available():
        return False
    from torch.distributed._tensor import DTensor

    return inspect.isfunction(value) and value is DTensor.from_local


class PlacementClassVariable(DistributedVariable):
    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    @staticmethod
    def is_placement_type(value):
        # we can't rely on importing/accessing torch distributed, it is not always built.
        if not DistributedVariable.is_available():
            return False

        from torch.distributed._tensor.placement_types import Placement

        return type(value) is type and issubclass(value, Placement)

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        options = VariableTracker.propagate(self, args, kwargs.values())
        if (
            inspect.getattr_static(self.value, "__new__", None) in (object.__new__,)
            and self.source
        ):
            # NOTE: we don't need to track mutations to the placement class as they
            # suppose to be immutable.
            new_obj = object.__new__(self.value)
            var = PlacementVariable(new_obj, **options)
            if inspect.getattr_static(self.value, "__init__", None):
                return var.add_options(var.call_method(tx, "__init__", args, kwargs))

        return super().call_function(tx, args, kwargs)


class PlacementVariable(DistributedVariable):
    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    @staticmethod
    def is_placement(value):
        # we can't rely on importing/accessing torch distributed, it is not always built.
        if not DistributedVariable.is_available():
            return False

        from torch.distributed._tensor.placement_types import Placement

        return istype(value, Placement)

    def as_python_constant(self):
        return self.value

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        from . import ConstantVariable

        options = VariableTracker.propagate(self, args, kwargs.values())
        allowed_methods = ["__init__", "__setattr__"]
        # placement types dynamo tracking allows only __init__
        # and __setattr__ methods, the latter is for case like `Shard(dim)`
        if name in allowed_methods:
            try:
                value_type = type(self.value)
                assert (
                    inspect.getattr_static(value_type, "__getattr__", None) is None
                ), "no custom getattr allowed!"
                method = inspect.getattr_static(value_type, name)
            except AttributeError:
                method = None
            if method is object.__init__:
                return ConstantVariable(None, **options)

            args = [x.as_python_constant() for x in args]
            kwargs = {k: v.as_python_constant() for k, v in kwargs.items()}
            method(self.value, *args, **kwargs)
            return self

        return super().call_method(tx, name, args, kwargs)


class DeviceMeshVariable(DistributedVariable):
    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    @staticmethod
    def is_device_mesh(value):
        # we can't rely on importing/accessing torch distributed, it is not always built.
        if not DistributedVariable.is_available():
            return False

        from torch.distributed._tensor.device_mesh import DeviceMesh

        return istype(value, DeviceMesh)

    def as_python_constant(self):
        return self.value
