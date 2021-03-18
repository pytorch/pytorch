import torch
import torch.fx
import inspect
import operator
import numbers
from typing import Any, Dict, Optional, Tuple
from torch.fx.node import Argument, Target
from torch._jit_internal import boolean_dispatched

from torch.fx import Transformer
from torch.fx.operator_schemas import get_signature_for_torch_op
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
        new_kwargs = None

        if self.normalize_functionals and target.__module__ == 'torch.nn.functional':
            target_for_analysis = target
            if target in boolean_dispatched:
                # HACK: `boolean_dispatch` as used in `torch.nn.functional` makes it so that we have
                # a 2-way dispatch based on a boolean value. Here we check that the `true` and `false`
                # branches of the dispatch have exactly the same signature. If they do, use the `true`
                # branch signature for analysis. Otherwise, leave this un-normalized
                assert not isinstance(target, str)
                dispatched = boolean_dispatched[target]
                if_true, if_false = dispatched['if_true'], dispatched['if_false']
                if inspect.signature(if_true).parameters != inspect.signature(if_false).parameters:
                    return super().call_function(target, args, kwargs)
                target_for_analysis = if_true

            assert callable(target_for_analysis)
            sig = inspect.signature(inspect.unwrap(target_for_analysis))
            new_kwargs = self._args_kwargs_to_normalized_kwargs(sig, args, kwargs)
        else:
            assert callable(target)
            torch_op_schemas = get_signature_for_torch_op(target)
            if torch_op_schemas:
                # Iterate through all of the schema until we find one that matches
                # If one matches, populate `new_kwargs` with the combined args/kwargs
                # values. If none matches, `new_kwargs` will be None
                for candidate_signature in torch_op_schemas:
                    try:
                        candidate_signature.bind(args, kwargs)
                        new_kwargs = self._args_kwargs_to_normalized_kwargs(candidate_signature, args, kwargs)
                        break
                    except TypeError:
                        continue
        if new_kwargs:
            # FIXME: `target(**kwargs)` doesn't keep things specified as kwargs
            # in kwargs
            return self.tracer.create_proxy('call_function', target, (), new_kwargs)
        else:
            return super().call_function(target, args, kwargs)

    def call_module(self, target : Target, args : Tuple[Argument, ...], kwargs : Dict[str, Any]):
        assert isinstance(target, str)
        submod = self.fetch_attr(target)
        if self.normalize_modules and hasattr(submod.__class__, '__name__'):
            classname = submod.__class__.__name__
            if getattr(torch.nn, classname, None) == submod.__class__:
                sig = inspect.signature(inspect.unwrap(submod.forward))
                new_kwargs = self._args_kwargs_to_normalized_kwargs(sig, args, kwargs)
                if new_kwargs:
                    return super().call_module(target, (), new_kwargs)
        return super().call_module(target, args, kwargs)

    def _args_kwargs_to_normalized_kwargs(self, sig : inspect.Signature, args : Tuple[Argument, ...],
                                          kwargs : Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Given a call target, args, and kwargs, return the arguments normalized into
        a single kwargs dict, or None if the type signature is not supported by
        this normalization.

        Args:

            target (inspect.Signature): Signature object for the target
            args (Tuple): Arguments that appear at the callsite for `target`
            kwargs (Dict): Keyword arugments that appear at the callsite for `target`

        Returns:

            Optional[Dict]: Normalized kwargs for `target`, or `None` if this target is not
                supported
        """

        # Don't currently support positional-only
        # or varargs (*args, **kwargs) signatures
        supported_parameter_types = {
            inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
        if any(p.kind not in supported_parameter_types for p in sig.parameters.values()):
            return None

        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        new_kwargs : Dict[str, Any] = {}
        for param in sig.parameters:
            new_kwargs[param] = bound_args.arguments[param]

        return new_kwargs


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
    binary_magic_method_remap = {
        operator.add : torch.add,
        operator.mul : torch.mul,
        operator.sub : torch.sub,
        operator.truediv : torch.div,
        operator.floordiv: torch.floor_divide,
        operator.mod: torch.remainder,
        operator.eq: torch.eq,
        operator.ne: torch.ne,
        operator.lt: torch.lt,
        operator.le: torch.le,
        operator.gt: torch.gt,
        operator.ge: torch.ge,
    }

    def call_function(self, target : Target, args : Tuple[Argument, ...], kwargs : Dict[str, Any]):
        # FIXME: placeholders do not get `Tensor` type by default
        #
        # Normalize operators according to the magic methods implemented on tensors here:
        # https://github.com/pytorch/pytorch/blob/28c5d90b679c6b38bf4183ec99f16d933c2f1bcd/tools/autograd/templates/python_variable_methods.cpp#L1137

        if target in self.binary_magic_method_remap:
            assert len(args) == 2
            assert len(kwargs) == 0
            lhs, rhs = args

            # If lhs is a Tensor, we definitely dispatch into the `torch` function
            if isinstance(lhs, torch.fx.Proxy) and lhs.node.type is torch.Tensor:
                rv = self.binary_magic_method_remap[target](lhs, rhs)
                rv.node.type = torch.Tensor
                return rv

            # If rhs is a Tensor but lhs is not, we *may* dispatch into the `torch`
            # function. Note that this is tricky to determine, as the LHS can do one
            # of the following:
            # 1. Do its own custom behavior. In this case `lhs.__add__(rhs)` returns
            #    a value. We do not want to normalize in this situation
            # 2. Not have an `__add__` method. In this case we would dispatch into
            #    `rhs.__radd__` and in turn call the torch method
            # 3. Have an `__add__` method but throw NotImplementedError. This would
            #    cause `rhs.__radd__` dispatch and subsequently call into the `torch`
            #    method, but it is hard to reason about whether that will happen
            #    statically
            #
            # In this case, we account for scenario (2) as well as an allowed list
            # of known-good types that when used as lhs with a Tensor rhs, dispatch
            # to the tensor method (e.g. numbers)

            # These operators don't commute, i.e. we can't have a Tensor on
            # rhs and a non-tensor on LHS
            non_commutative_operators = {operator.mod}

            if target not in non_commutative_operators and not isinstance(lhs, torch.fx.Proxy):
                if not hasattr(type(lhs), '__add__'):
                    rv = self.binary_magic_method_remap[target](lhs, rhs)
                    rv.node.type = torch.Tensor
                    return rv

                if isinstance(lhs, numbers.Number):
                    rv = self.binary_magic_method_remap[target](lhs, rhs)
                    rv.node.type = torch.Tensor
                    return rv

        return super().call_function(target, args, kwargs)
