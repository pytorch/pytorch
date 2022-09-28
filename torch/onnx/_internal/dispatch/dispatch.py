"""Symbolic function dispatcher.

Implementation reference: torch/_ops.py
"""
import types
from typing import Callable, Sequence

from torch.onnx import errors
from torch.onnx._internal import jit_utils, registration


class _OpDomain(types.ModuleType):
    """A class for resolving operator in a domain."""

    def __init__(self, domain: str):
        super().__init__(f"torch.onnx.symbolics.{domain}")
        self.domain = domain

    def __getattr__(self, name: str) -> Callable:
        qualified_op_name = f"{self.domain}::{name}"

        def symbolic_fn(*args, **kwargs):
            return _dispatch_qualified_name(qualified_op_name, args, kwargs)

        setattr(self, name, symbolic_fn)
        return symbolic_fn


class _SymbolicFunctionDispatcher(types.ModuleType):
    """Dispatches to the symbolic function for the given domain and op."""

    def __init__(self):
        super().__init__("torch.onnx.symbolics")

    def __getattr__(self, name: str):
        op_domain = _OpDomain(name)
        setattr(self, name, op_domain)
        return op_domain

    def __call__(self, qualified_name: str) -> Callable:
        # TODO(justinchuby): Add docstring
        def symbolic_fn(*args, **kwargs):
            return _dispatch_qualified_name(qualified_name, args, kwargs)

        return symbolic_fn


def _dispatch_qualified_name(
    qualified_name: str, args: Sequence, kwargs: dict
) -> Callable:
    """Dispatches to the symbolic function for the given qualified op name and arguments."""
    context = args[0]
    assert isinstance(context, jit_utils.GraphContext)
    symbolic_fn = registration.registry.get_function_group(qualified_name)
    if symbolic_fn is None:
        raise errors.UnsupportedOperatorError(
            qualified_name,
            context.opset,
            None,
        )
    return symbolic_fn(*args, **kwargs)


# The dispatcher for the symbolic functions.
symbolics = _SymbolicFunctionDispatcher()
