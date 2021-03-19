import torch
import torch.fx
import inspect
from typing import Any, Dict, List, Optional, Tuple
from torch.fx.node import Argument, Target
from torch._jit_internal import boolean_dispatched

from torch.fx import Transformer
from torch.fx.operator_schemas import get_signature_for_torch_op

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
                signature = self._find_overload_with_type_check(torch_op_schemas, args, kwargs)
                if signature:
                    new_kwargs = self._args_kwargs_to_normalized_kwargs(signature, args, kwargs)

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

    def _find_overload_with_type_check(
            self, candidate_signatures : List[inspect.Signature], args : Tuple[Argument, ...],
            kwargs : Dict[str, Any]) -> Optional[inspect.Signature]:
        """
        Perform overload resolution on a list of schema given `args` and `kwargs
        """
        found_signature = None
        for candidate_signature in candidate_signatures:
            try:
                bound_args = candidate_signature.bind(*args, **kwargs)
                bound_args.apply_defaults()

                # Signature.bind does not check types. Do secondary checking here
                for k, parameter in candidate_signature.parameters.items():
                    bound_arg = bound_args.arguments[k]
                    if parameter.annotation is not inspect.Signature.empty:
                        # We're the only ones to generate these signatures, we don't use
                        # string type annotations, so we should be good here
                        assert not isinstance(parameter.annotation, str)

                        # Assuming that all proxied arguments are of Tensor type because that's the only
                        # thing __torch_function__ currently supports
                        if isinstance(bound_arg, torch.fx.Proxy):
                            if parameter.annotation is not torch.Tensor:
                                raise TypeError()
                        elif parameter.annotation is torch.Tensor:
                            # Python arg parser accepts int, float and complex for Tensor-typed
                            # parameters
                            # https://github.com/pytorch/pytorch/blob/19792b45dbf30b4555c4a87512e624cdd4aa6e4c/torch/csrc/utils/python_arg_parser.cpp#L1077-L1100?  # noqa
                            if type(bound_arg) not in {torch.Tensor, int, float, complex}:
                                raise TypeError()
                        elif not issubclass(type(bound_arg), parameter.annotation):
                            raise TypeError()
                found_signature = candidate_signature
                break
            except TypeError:
                continue
        return found_signature
