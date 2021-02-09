import torch
import torch.fx
import inspect
from typing import Any, Dict, Optional

from torch.fx import Transformer

class NormalizeArgs(Transformer):
    """
    Normalize arguments to Python targets. This means that
    `args/kwargs` will be matched up to the module/functional's
    signature and rewritten to exclusively kwargs in positional order.
    Also populates default values. Does not support positional-only
    parameters or varargs parameters (*args, **kwargs).
    """
    def __init__(self, module, normalize_functionals : bool = True,
                 normalize_modules : bool = True):
        super().__init__(module)
        self.normalize_functionals = normalize_functionals
        self.normalize_modules = normalize_modules

    def call_function(self, target, args, kwargs):
        if self.normalize_functionals and target.__module__ == 'torch.nn.functional':
            new_kwargs = self._args_kwargs_to_normalized_kwargs(target, args, kwargs)

            if new_kwargs:
                # FIXME: `target(**kwargs)` doesn't keep things specified as kwargs
                # in kwargs
                return self.tracer.create_proxy('call_function', target, (), new_kwargs)

        return super().call_function(target, args, kwargs)

    def call_module(self, target, args, kwargs):
        assert isinstance(target, str)
        submod = self.fetch_attr(target)
        if self.normalize_modules and hasattr(submod.__class__, '__name__'):
            classname = submod.__class__.__name__
            if getattr(torch.nn, classname, None) == submod.__class__:
                new_kwargs = self._args_kwargs_to_normalized_kwargs(submod.forward, args, kwargs)
                if new_kwargs:
                    return super().call_module(target, (), new_kwargs)
        return super().call_module(target, args, kwargs)

    def _args_kwargs_to_normalized_kwargs(self, target, args, kwargs) -> Optional[Dict[str, Any]]:
        sig = inspect.signature(target)
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
