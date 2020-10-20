#!/usr/bin/env python3
from typing import Any, TypeVar, Optional, Tuple, List, NamedTuple, Union
import textwrap
import torch
from torch._C import TupleType, OptionalType, ListType


T = TypeVar("T")

MAX_RAW_TENSOR_SIZE = 16


class InflatableArg(NamedTuple):
    value: Any
    fmt: str


def augment_model_with_bundled_inputs(
        model: torch.jit.ScriptModule,
        inputs: Optional[List[Tuple[Any, ...]]] = None,
        _receive_inflate_expr: Optional[List[str]] = None,  # For debugging.
) -> None:
    """Add bundled sample inputs to a model.

    Models with bundled inputs can be invoked in a uniform manner by
    benchmarking and code coverage tools.

    Augmented models will support the following methods:

      `get_all_bundled_inputs() -> List[Tuple[Any, ...]]`
        Returns a list of tuples suitable for passing to the model like
        `for inp in model.get_all_bundled_inputs(): model(*inp)`

      `get_num_bundled_inputs() -> int`
        Equivalent to `len(model.get_all_bundled_inputs())`,
        but slightly easier to call from C++.

      `run_on_bundled_input(idx: int) -> Any`
        Run the model on bundled input number `idx`

    Inputs can be specified in one of two ways:

      - The model can define `_generate_bundled_inputs`
        get_all_bundled_inputs will simply call this method
        and cache the value.
      - The `inputs` argument to this function can be a list of tuples,
        of the same form that will be returned by get_all_bundled_inputs.
        This function will attempt to optimize arguments so that (e.g.)
        arguments like `torch.zeros(1000)` will be represented compactly.
        Only top-level arguments will be optimized.
        Tensors in lists or tuples will not.
    """
    if not isinstance(model, torch.jit.ScriptModule):
        raise Exception("Only ScriptModule is supported.")

    forward_arg_types = [arg.type for arg in model.forward.schema.arguments[1:]]
    deflated_inputs_type: ListType = ListType(TupleType(forward_arg_types))
    inflated_inputs_type: OptionalType[ListType] = OptionalType(deflated_inputs_type)
    model._c._register_attribute("_bundled_inputs_deflated", deflated_inputs_type, [])
    model._c._register_attribute("_bundled_inputs_inflated", inflated_inputs_type, None)

    if hasattr(model, "_generate_bundled_inputs"):
        if inputs is not None:
            raise Exception(
                "inputs is not None, but _generate_bundled_inputs is already defined")
        # Model author already defined _generate_bundled_inputs.
    elif inputs is None:
        raise Exception(
            "inputs must be specified if _generate_bundled_inputs is not already defined")
    else:
        # Iterate over the inputs and args in each input.
        # Accumulate `deflated_inputs` as (possibly) compressed values
        # and `parts` to be joined into the expression that unpacks them.
        deflated_inputs = []
        parts = []
        for inp_idx, args in enumerate(inputs):
            deflated_args = []
            parts.append("(")
            for arg_idx, arg in enumerate(args):
                deflated, inflater = _inflate_expr(arg, f"deflated[{inp_idx}][{arg_idx}]")
                deflated_args.append(deflated)
                parts.append(f"    {inflater},")
            deflated_inputs.append(tuple(deflated_args))
            parts.append("),")
        parts.append("")
        expr = "\n".join(parts)
        # Back-channel return this expr for debugging.
        if _receive_inflate_expr is not None:
            _receive_inflate_expr.append(expr)
        model._bundled_inputs_deflated = deflated_inputs
        definition = textwrap.dedent("""
            def _generate_bundled_inputs(self):
                deflated = self._bundled_inputs_deflated
                return [
            {}
                ]
            """).format(expr)
        model.define(definition)

    # Define get_all_bundled_inputs that caches the generated inputs.
    model.define(textwrap.dedent("""
        def get_all_bundled_inputs(self):
            if self._bundled_inputs_inflated is None:
                self._bundled_inputs_inflated = self._generate_bundled_inputs()
            all_inputs = self._bundled_inputs_inflated
            assert all_inputs is not None
            return all_inputs
        """))

    # Define some helper methods.
    model.define(textwrap.dedent("""
        def get_num_bundled_inputs(self):
            return len(self.get_all_bundled_inputs())
        """))
    model.define(textwrap.dedent("""
        def run_on_bundled_input(self, idx: int):
            return self(*self.get_all_bundled_inputs()[idx])
        """))


def _inflate_expr(arg: T, ref: str) -> Tuple[Union[T, torch.Tensor], str]:
    # Allow custom inflation expressions any object.
    # For example, calling custom image-decoding ops.
    # Or just use "{}" as the format string to ignore size limits.
    if isinstance(arg, InflatableArg):
        return arg.value, arg.fmt.format(ref)

    if isinstance(arg, torch.Tensor):
        # Small-storage tensors can just be saved directly.
        if arg.storage().size() <= MAX_RAW_TENSOR_SIZE:
            return arg, ref
        # Small contiguous tensors can be cloned to have small storage.
        # TODO: Should we do this even for non-contiguous tensors?
        if arg.is_contiguous() and arg.numel() <= MAX_RAW_TENSOR_SIZE:
            return arg.clone(), ref
        # Example inputs commonly come from torch.zeros, torch.ones, or torch.full.
        # These can be represented compactly.
        for fmt in [torch.contiguous_format, torch.channels_last]:
            if arg.is_contiguous(memory_format=fmt) and (arg == arg.flatten()[0]).all().item():
                return (torch.tensor([arg.flatten()[0]]).expand(*arg.size()),
                        f"{ref}.contiguous(memory_format={fmt})")
        # Prevent big tensors from being bundled by default.
        # TODO: Provide more useful diagnostics.
        raise Exception(
            f"Bundled input argument at position '{ref}' is "
            f"a tensor with storage size {arg.storage().size()}. "
            f"You probably don't want to bundle this as an input. "
        )
    else:
        return arg, ref


def bundle_randn(*size, dtype=None):
    """Generate a tensor that will be inflated with torch.randn."""
    stub = torch.zeros(1, dtype=dtype).expand(*size)
    return InflatableArg(value=stub, fmt="torch.randn_like({})")


def bundle_large_tensor(t):
    """Wrap a tensor to allow bundling regardless of size."""
    return InflatableArg(value=t, fmt="{}")
