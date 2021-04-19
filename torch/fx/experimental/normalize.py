import torch
import torch.fx
import operator
from typing import Any, Callable, Dict, Tuple
from torch.fx.node import Argument, Target
from torch.fx.operator_schemas import normalize_module, normalize_function

from torch.fx import Transformer
from .schema_type_annotation import AnnotateTypesWithSchema

class NormalizeArgs(Transformer):
    """
    Normalize arguments to Python targets. This means that
    `args/kwargs` will be matched up to the module/functional's
    signature and rewritten to exclusively kwargs in positional order.
    Also populates default values. Does not support positional-only
    parameters or varargs parameters (*args, **kwargs).
    Example usage:
        m = torchvision.models.resnet18()
        traced = torch.fx.symbolic_trace(m)
        traced = NormalizeArgs(traced).transform()
    """
    def __init__(self, module : torch.nn.Module, normalize_functionals : bool = True,
                 normalize_modules : bool = True):
        super().__init__(module)
        self.normalize_functionals = normalize_functionals
        self.normalize_modules = normalize_modules

    def call_function(self, target : Target, args : Tuple[Argument, ...], kwargs : Dict[str, Any]):
        assert callable(target)
        new_kwargs = normalize_function(target, args, kwargs)  # type: ignore
        if new_kwargs:
            return self.tracer.create_proxy('call_function', target, (), new_kwargs)
        else:
            return super().call_function(target, args, kwargs)

    def call_module(self, target : Target, args : Tuple[Argument, ...], kwargs : Dict[str, Any]):
        assert isinstance(target, str)
        new_kwargs = normalize_module(self.module, target, args, kwargs)  # type: ignore
        if new_kwargs:
            return super().call_module(target, (), new_kwargs)
        else:
            return super().call_module(target, args, kwargs)

class NormalizeOperators(AnnotateTypesWithSchema):
    """
    Normalize callsites that are different ways of "spelling" the same
    invocation into a single, canonical call. Currently supports:

    1. Normalize operators (e.g. operator.add) to the `torch` ops they
       ultimately invoke (e.g. torch.add) when it is possible to statically
       reason that

    Example usage:

        m = torchvision.models.resnet18()

        traced = torch.fx.symbolic_trace(m)

        traced = NormalizeOperators(traced).transform()
    """
    binary_magic_method_remap : Dict[Callable[[Any, Any], Any], Callable[[Any, Any], Any]] = {
        torch.add : operator.add,
        torch.mul : operator.mul,
        torch.sub : operator.sub,
        torch.div : operator.truediv,
        torch.floor_divide : operator.floordiv,
        torch.remainder : operator.mod,
        torch.eq : operator.eq,
        torch.ne : operator.ne,
        torch.lt : operator.lt,
        torch.le : operator.le,
        torch.gt : operator.gt,
        torch.ge : operator.ge,
    }

    def call_function(self, target : Target, args : Tuple[Argument, ...], kwargs : Dict[str, Any]):
        # Normalize operators according to the magic methods implemented on tensors here:
        # https://github.com/pytorch/pytorch/blob/28c5d90b679c6b38bf4183ec99f16d933c2f1bcd/tools/autograd/templates/python_variable_methods.cpp#L1137 # noqa: B950

        assert callable(target)

        if target in self.binary_magic_method_remap:
            if len(args) != 2:
                return super().call_function(target, args, kwargs)
            lhs, rhs = args

            return super().call_function(
                target=self.binary_magic_method_remap[target], args=(lhs, rhs), kwargs={})

        return super().call_function(target, args, kwargs)
