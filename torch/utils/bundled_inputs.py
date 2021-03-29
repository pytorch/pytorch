#!/usr/bin/env python3
from typing import Any, TypeVar, Optional, Tuple, List, NamedTuple, Union, Sequence, Dict, Callable
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
        inputs: Optional[Sequence[Tuple[Any, ...]]] = None,
        _receive_inflate_expr: Optional[List[str]] = None,  # For debugging.
        info: Optional[List[str]] = None,  # Optional argument to provide info about forward or its inputs
) -> None:
    """ Wrapper around augment_many_model_functions_with_bundled_inputs to provide a streamlined api for forward
    which is the only function the vast majority of models need bundled inputs for.
    """

    if not isinstance(model, torch.jit.ScriptModule):
        raise Exception("Only ScriptModule is supported.")

    forward: Callable = model.forward

    # Sometimes forward won't have a name attached so just in case
    if not hasattr(forward, "__name__"):
        forward.__name__ = 'forward'
    augment_many_model_functions_with_bundled_inputs(
        model,
        inputs={forward : inputs},
        _receive_inflate_expr=_receive_inflate_expr,
        info={forward : info} if info else None,
    )


def augment_many_model_functions_with_bundled_inputs(
        model: torch.jit.ScriptModule,
        inputs: Dict[Callable, Optional[Sequence[Tuple[Any, ...]]]],
        _receive_inflate_expr: Optional[List[str]] = None,  # For debugging.
        info: Optional[Dict[Callable, List[str]]] = None,  # Optional argument to provide info about the function or its inputs
) -> None:
    """Add bundled sample inputs to a model for an arbitrary list of public functions.

    Models with bundled inputs can be invoked in a uniform manner by
    benchmarking and code coverage tools.

    Augmented models will support the following methods:

        `get_all_bundled_inputs_for_<function_name>() -> List[Tuple[Any, ...]]`
            Returns a list of tuples suitable for passing to the model like
            `for inp in model.get_all_bundled_inputs_for_foo(): model.foo(*inp)`

        `get_bundled_inputs_functions_and_info() -> Dict[str, Dict[str: List[str]]]`
            Returns a dictionary mapping function names to a metadata dictionary.
            This nested dictionary maps preset strings like:
                'get_inputs_function_name' -> the name of a function attribute in this model that can be
                    run to get back a list of inputs corresponding to that function.
                'info' -> the user provided extra information about the bundled inputs

    If forward has bundled inputs then these following functions are also defined:

        `get_all_bundled_inputs() -> List[Tuple[Any, ...]]`
            Returns a list of tuples suitable for passing to the model like
            `for inp in model.get_all_bundled_inputs(): model(*inp)`

        `get_num_bundled_inputs() -> int`
            Equivalent to `len(model.get_all_bundled_inputs())`,
            but slightly easier to call from C++.

        `run_on_bundled_input(idx: int) -> Any`
            Run the model on bundled input number `idx`

    Inputs can be specified in one of two ways:

      - The model can define `_generate_bundled_inputs_for_<function_name>`
        get_all_bundled_inputs will simply call this method
        and cache the value. If the user chooses this method inputs[<function>]
        should map to None
      - The `inputs` argument to this function can be a dictionary mapping functions to a
        list of inputs, of the same form that will be returned by get_all_bundled_inputs_for_<function_name>.
        The type of the inputs is List[Tuple[Any, ...]]. The outer list corresponds with a
        list of inputs, the inner tuple is the list of args that together make up one input.
        For inputs of functions that take one arg, this will be a tuple of length one. The Any, ...
        is the actual data that makes up the args, e.g. a tensor.

    Info is an optional parameter that maps functions to a list of strings providing extra information about that
    function's bundled inputs. This could be descriptions, expected outputs, etc.
        - Ex: info={model.forward : ['man eating icecream', 'an airplane', 'a dog']}

    This function will attempt to optimize arguments so that (e.g.)
    arguments like `torch.zeros(1000)` will be represented compactly.
    Only top-level arguments will be optimized.
    Tensors in lists or tuples will not.
    """
    if not isinstance(model, torch.jit.ScriptModule):
        raise Exception("Only ScriptModule is supported.")

    if not inputs:
        raise Exception("Please provide inputs for at least 1 function")

    get_bundled_inputs_functions_and_info_template = ""

    for function, input_list in inputs.items():
        function_name = function.__name__

        if input_list is not None and not isinstance(input_list, Sequence):
            raise TypeError("Error inputs for function {0} is not a Sequence".format(function_name))

        function_arg_types = [arg.type for arg in function.schema.arguments[1:]]  # type: ignore
        deflated_inputs_type: ListType = ListType(TupleType(function_arg_types))
        inflated_inputs_type: OptionalType[ListType] = OptionalType(deflated_inputs_type)
        model._c._register_attribute("_bundled_inputs_deflated_{name}".format(name=function_name), deflated_inputs_type, [])
        model._c._register_attribute("_bundled_inputs_inflated_{name}".format(name=function_name), inflated_inputs_type, None)

        if hasattr(model, "_generate_bundled_inputs_for_" + function_name):
            if input_list is not None:
                raise Exception(
                    "inputs[{name}] is not None, but _generate_bundled_inputs_for_{name} is already defined".format(
                        name=function_name
                    )
                )
            # Model author already defined _generate_bundled_inputs_for_<function_name>.
        elif input_list is None or len(input_list) == 0:
            raise Exception(
                "inputs for {name} must be specified if _generate_bundled_inputs_for_{name} is not already defined".format(
                    name=function_name,
                )
            )
        else:
            # Iterate over the inputs and args in each input.
            # Accumulate `deflated_inputs` as (possibly) compressed values
            # and `parts` to be joined into the expression that unpacks them.
            deflated_inputs = []
            parts = []
            for inp_idx, args in enumerate(input_list):
                if not isinstance(args, Tuple) and not isinstance(args, List):  # type: ignore
                    raise TypeError(
                        "Error bundled input for function {0} idx: {1} is not a Tuple or a List".format(function_name, inp_idx)
                    )
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
            setattr(model, "_bundled_inputs_deflated_{name}".format(name=function_name), deflated_inputs)
            definition = textwrap.dedent("""
                def _generate_bundled_inputs_for_{name}(self):
                    deflated = self._bundled_inputs_deflated_{name}
                    return [
                {expr}
                    ]
                """).format(expr=expr, name=function_name)
            model.define(definition)

        # Define get_all_bundled_inputs_for_<function_name> that caches the generated inputs.
        model.define(textwrap.dedent("""
            def get_all_bundled_inputs_for_{name}(self):
                if self._bundled_inputs_inflated_{name} is None:
                    self._bundled_inputs_inflated_{name} = self._generate_bundled_inputs_for_{name}()
                all_inputs = self._bundled_inputs_inflated_{name}
                assert all_inputs is not None
                return all_inputs
            """).format(name=function_name))

        # Add to the high level helper methods
        inputs_info = repr(info[function]) if info and function in info else '[]'
        get_bundled_inputs_functions_and_info_template += """
            temp_dict : Dict[str,List[str]] = {{}}
            info: List[str] = {info}

            temp_dict['info'] = info
            temp_dict['get_inputs_function_name'] = ['get_all_bundled_inputs_for_{name}']
            all_inputs['{name}'] = temp_dict
            """.format(
            name=function_name,
            info=inputs_info,
        )

        # To ensure backwards compatibility and a streamlined api for forward these wrappers are provided
        if function_name == 'forward':
            model.define(textwrap.dedent("""
                def get_all_bundled_inputs(self):
                    return self.get_all_bundled_inputs_for_forward()
                """))
            model.define(textwrap.dedent("""
                def get_num_bundled_inputs(self):
                    return len(self.get_all_bundled_inputs_for_forward())
                """))
            model.define(textwrap.dedent("""
                def run_on_bundled_input(self, idx: int):
                    return self(*self.get_all_bundled_inputs()[idx])
                """))


    # Define some high level helper methods that act on all bundled inputs
    model.define(textwrap.dedent("""
        def get_bundled_inputs_functions_and_info(self):
            all_inputs : Dict[str, Dict[str,List[str]]] = {{}}
            {template}
            return all_inputs
        """.format(template=get_bundled_inputs_functions_and_info_template)))

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
