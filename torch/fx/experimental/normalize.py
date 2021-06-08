import torch
import torch.fx
import torch.fx as fx
import operator
from typing import Any, Callable, Dict, Tuple, Optional
from torch.fx.node import Argument, Target, Node, map_aggregate
from torch.fx.operator_schemas import normalize_module, normalize_function, create_type_hint

from torch.fx import Transformer, Proxy
from .schema_type_annotation import AnnotateTypesWithSchema

class NormalizeArgs(Transformer):
    """
    Normalize arguments to Python targets. This means that
    `args/kwargs` will be matched up to the module/functional's
    signature and rewritten to exclusively kwargs in positional order
    if `normalize_to_only_use_kwargs` is true. Also populates default
    values. Does not support positional-only parameters or varargs
    parameters (*args, **kwargs).

    If the nodes have 'type' metadata, it will use it to disambiguate
    overloads. Otherwise, it will throw an error.

    Example usage:
        m = torchvision.models.resnet18()
        traced = torch.fx.symbolic_trace(m)
        traced = NormalizeArgs(traced).transform()
    """
    def __init__(self, module : torch.nn.Module,
                 normalize_to_only_use_kwargs : bool = True):
        super().__init__(module)
        self.node_map: Dict[Proxy, Node] = {}
        self.normalize_to_only_use_kwargs = normalize_to_only_use_kwargs

    def run_node(self, n: Node) -> Any:
        args, kwargs = self.fetch_args_kwargs_from_env(n)

        def get_type(arg):
            if isinstance(arg, fx.Node):
                return n.meta['type'] if 'type' in n.meta else None
            return type(arg)

        arg_types = map_aggregate(n.args, get_type)
        assert(isinstance(arg_types, tuple))
        arg_types = tuple([create_type_hint(i) for i in arg_types])
        kwarg_types = {k: get_type(v) for k, v in kwargs.items()}
        if n.op == 'call_function':
            out = self.call_function(n.target, args, kwargs, arg_types, kwarg_types)
        else:
            out = super().run_node(n)
        if n.op != "output":
            self.node_map[out] = n
        return out

    def call_function(
            self, target : Target, args : Tuple[Argument, ...], kwargs : Dict[str, Any],
            arg_types: Optional[Tuple[Any, ...]] = None, kwarg_types : Optional[Dict[str, Any]] = None):
        assert callable(target)
        new_args_and_kwargs = normalize_function(target, args, kwargs, arg_types, kwarg_types,  # type: ignore[arg-type]
                                                 self.normalize_to_only_use_kwargs)
        if new_args_and_kwargs:
            new_args, new_kwargs = new_args_and_kwargs
            return self.tracer.create_proxy('call_function', target, new_args, new_kwargs)
        else:
            return super().call_function(target, args, kwargs)

    def call_module(self, target : Target, args : Tuple[Argument, ...], kwargs : Dict[str, Any]):
        assert isinstance(target, str)
        new_args_and_kwargs = normalize_module(self.module, target, args, kwargs,  # type: ignore[arg-type]
                                               self.normalize_to_only_use_kwargs)
        if new_args_and_kwargs:
            new_args, new_kwargs = new_args_and_kwargs
            return super().call_module(target, new_args, new_kwargs)
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
