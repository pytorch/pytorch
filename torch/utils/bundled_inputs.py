#!/usr/bin/env python3
from typing import Any, TypeVar, Optional, Tuple, List, NamedTuple, Union, Sequence, Dict, Callable
import textwrap
import torch
from torch._C import TupleType, ListType
from torch.jit._recursive import wrap_cpp_module


T = TypeVar("T")

MAX_RAW_TENSOR_SIZE = 16

class InflatableArg(NamedTuple):
    """ Helper type for bundled inputs.

        'value' is the compressed/deflated input that is stored in the model. Value
        must be of the same type as the argument to the function that it is a deflated
        input for.

        'fmt' is a formatable code string that is executed to inflate the compressed data into
        the appropriate input. It can use 'value' as an input to the format str. It must result
        in a value of the same type as 'value'.

        'fmt_fn' is a formatable function code string that is executed to inflate the compressed
        data into the appropriate input. It must result in a value of the same type as 'value'.
        The function name should be the formatable part of the string.

    Note: Only top level InflatableArgs can be inflated. i.e. you cannot place
    an inflatable arg inside of some other structure. You should instead create
    an inflatable arg such that the fmt code string returns the full structure
    of your input.
    """
    value: Any
    fmt: str = "{}"
    fmt_fn: str = ""


def bundle_inputs(
        model: torch.jit.ScriptModule,
        inputs: Union[Optional[Sequence[Tuple[Any, ...]]], Dict[Callable, Optional[Sequence[Tuple[Any, ...]]]]],
        info: Optional[Union[List[str], Dict[Callable, List[str]]]] = None,
        *,
        _receive_inflate_expr: Optional[List[str]] = None,
) -> torch.jit.ScriptModule:
    """Creates and returns a copy of the specified model with inputs attached. The original model is
    not mutated or changed in any way.

    Models with bundled inputs can be invoked in a uniform manner by
    benchmarking and code coverage tools.

    If inputs is passed in as a list then the inputs will be bundled for 'forward'.
    If inputs is instead passed in as a map then all the methods specified in the map
    will have their corresponding inputs bundled. Info should match watchever type is
    chosen for the inputs.

    The returned model will support the following methods:

        `get_all_bundled_inputs_for_<function_name>() -> List[Tuple[Any, ...]]`
            Returns a list of tuples suitable for passing to the model like
            `for inp in model.get_all_bundled_inputs_for_foo(): model.foo(*inp)`

        `get_bundled_inputs_functions_and_info() -> Dict[str, Dict[str: List[str]]]`
            Returns a dictionary mapping function names to a metadata dictionary.
            This nested dictionary maps preset strings like:
                'get_inputs_function_name' -> the name of a function attribute in this model that can be
                    run to get back a list of inputs corresponding to that function.
                'info' -> the user provided extra information about the bundled inputs

    If forward has bundled inputs then these following functions will also be defined on the returned module:

        `get_all_bundled_inputs() -> List[Tuple[Any, ...]]`
            Returns a list of tuples suitable for passing to the model like
            `for inp in model.get_all_bundled_inputs(): model(*inp)`

        `get_num_bundled_inputs() -> int`
            Equivalent to `len(model.get_all_bundled_inputs())`,
            but slightly easier to call from C++.

    Inputs can be specified in one of two ways:

      - The model can define `_generate_bundled_inputs_for_<function_name>`.
        If the user chooses this method inputs[<function>] should map to None

      - The `inputs` argument to this function can be a dictionary mapping functions to a
        list of inputs, of the same form that will be returned by get_all_bundled_inputs_for_<function_name>.
        Alternatively if only bundling inputs for forward the map can be omitted and a singular list of inputs
        can be provided instead.

        The type of the inputs is List[Tuple[Any, ...]]. The outer list corresponds with a
        list of inputs, the inner tuple is the list of args that together make up one input.
        For inputs of functions that take one arg, this will be a tuple of length one. The Any, ...
        is the actual data that makes up the args, e.g. a tensor.

    Info is an optional parameter that maps functions to a list of strings providing extra information about that
    function's bundled inputs. Alternatively if only bundling inputs for forward the map can be omitted and
    a singular list of information can be provided instead. This could be descriptions, expected outputs, etc.
        - Ex: info={model.forward : ['man eating icecream', 'an airplane', 'a dog']}

    This function will attempt to optimize arguments so that (e.g.)
    arguments like `torch.zeros(1000)` will be represented compactly.
    Only top-level arguments will be optimized.
    Tensors in lists or tuples will not.
    """
    if not isinstance(model, torch.jit.ScriptModule):
        raise Exception("Only ScriptModule is supported.")

    ignored_methods, ignored_attrs = _get_bundled_inputs_attributes_and_methods(model)
    clone = torch._C._hack_do_not_use_clone_module_with_class(  # type: ignore[attr-defined]
        model._c,
        ignored_methods,
        ignored_attrs,
    )

    # The above cloning function returns a torch._C.scriptmodule and we need a torch.jit.scriptmodule.
    # Fortunately theres a function in _recursive that does exactly that conversion.
    cloned_module = wrap_cpp_module(clone)
    if isinstance(inputs, dict):
        assert(isinstance(info, dict) or info is None)
        augment_many_model_functions_with_bundled_inputs(cloned_module, inputs, _receive_inflate_expr, info)
    else:
        assert(isinstance(info, list) or info is None)
        augment_model_with_bundled_inputs(cloned_module, inputs, _receive_inflate_expr, info)
    return cloned_module

def augment_model_with_bundled_inputs(
        model: torch.jit.ScriptModule,
        inputs: Optional[Sequence[Tuple[Any, ...]]] = None,
        _receive_inflate_expr: Optional[List[str]] = None,  # For debugging.
        info: Optional[List[str]] = None,  # Optional argument to provide info about forward or its inputs
        skip_size_check=False,
) -> None:
    """ Add bundled sample inputs to a model for the forward function.

    Models with bundled inputs can be invoked in a uniform manner by
    benchmarking and code coverage tools.

    Augmented models will support the following methods:

        `get_all_bundled_inputs() -> List[Tuple[Any, ...]]`
            Returns a list of tuples suitable for passing to the model like
            `for inp in model.get_all_bundled_inputs(): model(*inp)`

        `get_num_bundled_inputs() -> int`
            Equivalent to `len(model.get_all_bundled_inputs())`,
            but slightly easier to call from C++.

        `get_bundled_inputs_functions_and_info() -> Dict[str, Dict[str: List[str]]]`
            Returns a dictionary mapping function names to a metadata dictionary.
            This nested dictionary maps preset strings like:
                'get_inputs_function_name' -> the name of a function attribute in this model that can be
                    run to get back a list of inputs corresponding to that function.
                'info' -> the user provided extra information about the bundled inputs

    Inputs can be specified in one of two ways:

      - The model can define `_generate_bundled_inputs_for_forward`.
        If the user chooses this method inputs should be None

      - `inputs` is a list of inputs of form List[Tuple[Any, ...]]. A list of tuples where the elements
        of each tuple are the args that make up one input.
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
        skip_size_check=skip_size_check,
    )


def augment_many_model_functions_with_bundled_inputs(
        model: torch.jit.ScriptModule,
        inputs: Dict[Callable, Optional[Sequence[Tuple[Any, ...]]]],
        _receive_inflate_expr: Optional[List[str]] = None,  # For debugging.
        info: Optional[Dict[Callable, List[str]]] = None,  # Optional argument to provide info about the function or its inputs
        skip_size_check=False,
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

    Inputs can be specified in one of two ways:

      - The model can define `_generate_bundled_inputs_for_<function_name>`.
        If the user chooses this method inputs[<function>] should map to None

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

    if hasattr(model, "get_all_bundled_inputs") or hasattr(model, "get_bundled_inputs_functions_and_info"):
        raise Exception(
            "Models can only be augmented with bundled inputs once. "
            "This Model seems to have already been augmented with "
            "bundled inputs. Please start afresh with one that "
            "doesn't have bundled inputs.",
        )

    get_bundled_inputs_functions_and_info_template = ""

    for function, input_list in inputs.items():
        if hasattr(function, "__name__"):
            function_name = function.__name__
        else:
            if hasattr(function, "name"):
                function_name = function.name  # type: ignore[attr-defined]
            else:
                raise Exception(
                    'At least one of your functions has no attribute name please ensure all have one. m.foo.name = "foo"')


        if input_list is not None and not isinstance(input_list, Sequence):
            raise TypeError(f"Error inputs for function {function_name} is not a Sequence")

        function_arg_types = [arg.type for arg in function.schema.arguments[1:]]  # type: ignore[attr-defined]
        deflated_inputs_type: ListType = ListType(TupleType(function_arg_types))
        model._c._register_attribute(f"_bundled_inputs_deflated_{function_name}", deflated_inputs_type, [])

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
                if not isinstance(args, Tuple) and not isinstance(args, List):  # type: ignore[arg-type]
                    raise TypeError(
                        f"Error bundled input for function {function_name} idx: {inp_idx} is not a Tuple or a List"
                    )
                deflated_args = []
                parts.append("(")
                for arg_idx, arg in enumerate(args):
                    inflate_helper_fn_name = _get_inflate_helper_fn_name(arg_idx, inp_idx, function_name)
                    deflated, inflater, helper_definition = _inflate_expr(
                        arg,
                        f"deflated[{inp_idx}][{arg_idx}]",
                        inflate_helper_fn_name,
                        skip_size_check=skip_size_check,
                    )
                    deflated_args.append(deflated)
                    parts.append(f"    {inflater},")
                    if helper_definition:
                        model.define(textwrap.dedent(helper_definition))
                deflated_inputs.append(tuple(deflated_args))
                parts.append("),")
            parts.append("")
            expr = "\n".join(parts)

            # Back-channel return this expr for debugging.
            if _receive_inflate_expr is not None:
                _receive_inflate_expr.append(expr)
            setattr(model, f"_bundled_inputs_deflated_{function_name}", deflated_inputs)
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
                all_inputs = self._generate_bundled_inputs_for_{name}()
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

    # Define some high level helper methods that act on all bundled inputs
    model.define(textwrap.dedent("""
        def get_bundled_inputs_functions_and_info(self):
            all_inputs : Dict[str, Dict[str,List[str]]] = {{}}
            {template}
            return all_inputs
        """.format(template=get_bundled_inputs_functions_and_info_template)))

def _inflate_expr(
    arg: T, ref: str, inflate_helper_fn_name: str, skip_size_check: bool = False
) -> Tuple[Union[T, torch.Tensor], str, Optional[str]]:
    # Allow custom inflation expressions any object.
    # For example, calling custom image-decoding ops.
    # Or just use "{}" as the format string to ignore size limits.
    if isinstance(arg, InflatableArg):
        if arg.fmt_fn:
            if arg.fmt not in ["{}", ""]:
                raise Exception(
                    f"Bundled input argument at position '{ref}' has "
                    f"both arg.fmt_fn => \n{arg.fmt_fn} "
                    f"\n and arg.fmt  => {arg.fmt}. "
                    "Please choose `arg.fmt` if the deflater is straightforward or "
                    "`arg.fmt_fn` if you need a function."
                )

            helper_definition = arg.fmt_fn.format(inflate_helper_fn_name)
            expr = f"self.{inflate_helper_fn_name}({ref})"

            return arg.value, expr, helper_definition
        else:
            return arg.value, arg.fmt.format(ref), None

    if isinstance(arg, torch.Tensor):
        # Small-storage tensors can just be saved directly.
        if arg._typed_storage().size() <= MAX_RAW_TENSOR_SIZE or skip_size_check:
            return arg, ref, None
        # Small contiguous tensors can be cloned to have small storage.
        # TODO: Should we do this even for non-contiguous tensors?
        if arg.is_contiguous() and arg.numel() <= MAX_RAW_TENSOR_SIZE:
            return arg.clone(), ref, None
        # Example inputs commonly come from torch.zeros, torch.ones, or torch.full.
        # These can be represented compactly.
        for fmt in [torch.contiguous_format, torch.channels_last]:
            if arg.is_contiguous(memory_format=fmt) and (arg == arg.flatten()[0]).all().item():
                return (arg.flatten()[0].clone().expand(*arg.size()),
                        f"{ref}.contiguous(memory_format={fmt})", None)
        # Prevent big tensors from being bundled by default.
        # TODO: Provide more useful diagnostics.
        raise Exception(
            f"Bundled input argument at position '{ref}' is "
            f"a tensor with storage size {arg._typed_storage().size()}. "
            f"You probably don't want to bundle this as an input. "
        )
    else:
        return arg, ref, None

def _get_bundled_inputs_attributes_and_methods(script_module: torch.jit.ScriptModule) -> Tuple[List[str], List[str]]:
    methods: List[str] = []
    attributes: List[str] = []

    # Has bundled inputs for forward
    if hasattr(script_module, 'get_all_bundled_inputs'):
        methods.append('get_all_bundled_inputs')
        methods.append('get_num_bundled_inputs')
        methods.append('run_on_bundled_input')

    if hasattr(script_module, 'get_bundled_inputs_functions_and_info'):
        methods.append('get_bundled_inputs_functions_and_info')
        all_info = script_module.get_bundled_inputs_functions_and_info()
        for function_name in all_info:
            methods.append("get_all_bundled_inputs_for_" + function_name)
            methods.append("_generate_bundled_inputs_for_" + function_name)
            attributes.append("_bundled_inputs_deflated_" + function_name)

            bundled_inputs_fn = getattr(
                script_module,
                f"get_all_bundled_inputs_for_{function_name}"
            )
            num_bundled_inputs: int = len(bundled_inputs_fn())

            # Check inflate helper functions for each function, argument and bundled input
            func = getattr(script_module, function_name)
            for arg_idx in range(len(func.schema.arguments) - 1):
                for input_idx in range(num_bundled_inputs):
                    helper_fn_name = _get_inflate_helper_fn_name(
                        arg_idx=arg_idx,
                        input_idx=input_idx,
                        function_name=function_name
                    )
                    # if the arg has an InflatableArg with fmt_fn, add the helper function name
                    if hasattr(script_module, helper_fn_name):
                        methods.append(helper_fn_name)

    return (methods, attributes)


def _get_inflate_helper_fn_name(
    arg_idx: int,
    input_idx: int,
    function_name: str,
) -> str:
    return f"_inflate_helper_for_{function_name}_input_{input_idx}_arg_{arg_idx}"



def bundle_randn(*size, dtype=None):
    """Generate a tensor that will be inflated with torch.randn."""
    stub = torch.zeros(1, dtype=dtype).expand(*size)
    return InflatableArg(value=stub, fmt="torch.randn_like({})")


def bundle_large_tensor(t):
    """Wrap a tensor to allow bundling regardless of size."""
    return InflatableArg(value=t, fmt="{}")
