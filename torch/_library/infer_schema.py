# mypy: allow-untyped-defs
import collections
import inspect
import typing
from dataclasses import dataclass
from types import GenericAlias
from typing import Any, Optional, Union

import torch
from torch import device, dtype, Tensor, types
from torch.utils._exposed_in import exposed_in


# This is used as a negative test for
# test_custom_ops.py::TestTypeConversion::test_type_eval.
_TestTensor = torch.Tensor


@exposed_in("torch.library")
def infer_schema(
    prototype_function: typing.Callable,
    /,
    *,
    mutates_args,
    op_name: Optional[str] = None,
) -> str:
    r"""Parses the schema of a given function with type hints. The schema is inferred from the
    function's type hints, and can be used to define a new operator.

    We make the following assumptions:

    * None of the outputs alias any of the inputs or each other.
    * | String type annotations "device, dtype, Tensor, types" without library specification are
      | assumed to be torch.*. Similarly, string type annotations "Optional, List, Sequence, Union"
      | without library specification are assumed to be typing.*.
    * | Only the args listed in ``mutates_args`` are being mutated. If ``mutates_args`` is "unknown",
      | it assumes that all inputs to the operator are being mutates.

    Callers (e.g. the custom ops API) are responsible for checking these assumptions.

    Args:
        prototype_function: The function from which to infer a schema for from its type annotations.
        op_name (Optional[str]): The name of the operator in the schema. If ``name`` is None, then the
            name is not included in the inferred schema. Note that the input schema to
            ``torch.library.Library.define`` requires a operator name.
        mutates_args ("unknown" | Iterable[str]): The arguments that are mutated in the function.

    Returns:
        The inferred schema.

    Example:
        >>> def foo_impl(x: torch.Tensor) -> torch.Tensor:
        >>>     return x.sin()
        >>>
        >>> infer_schema(foo_impl, op_name="foo", mutates_args={})
        foo(Tensor x) -> Tensor
        >>>
        >>> infer_schema(foo_impl, mutates_args={})
        (Tensor x) -> Tensor
    """
    UNKNOWN_MUTATES = "unknown"
    pf_globals = prototype_function.__globals__
    pf_locals = None
    # TODO: Once our minimum version is py3.10+ pass `eval_str=True` to
    # inspect.signature() and we no longer need to deal with stringified
    # annotations below.
    sig = inspect.signature(prototype_function)

    def error_fn(what):
        raise ValueError(f"infer_schema(func): {what} Got func with signature {sig})")

    def convert_type_string(annotation_type: str):
        try:
            return eval(annotation_type, pf_globals, pf_locals)
        except Exception:
            error_fn(
                f"Unsupported type annotation {annotation_type}. It is not a type."
            )

    def unstringify_types(
        tys: tuple[Union[type[object], str], ...]
    ) -> tuple[tuple[typing.Any, ...], bool]:
        res = []
        changed = False
        for ty in tys:
            ty, ty_changed = unstringify_type(ty)
            res.append(ty)
            changed |= ty_changed
        if changed:
            return tuple(res), True
        else:
            return tys, False  # type: ignore[return-value]

    def unstringify_type(ty: Union[type[object], str]) -> tuple[typing.Any, bool]:
        # Dig through a generic type and if it contains a stringified type
        # convert that to a real type. The second return value indicates if the
        # type contained a string or not.
        if isinstance(ty, str):
            return convert_type_string(ty), True
        elif origin := typing.get_origin(ty):
            args, args_changed = unstringify_types(typing.get_args(ty))
            if args_changed:
                return GenericAlias(origin, args), True

        return ty, False

    params = []
    seen_args = set()
    saw_kwarg_only_arg = False
    for idx, (name, param) in enumerate(sig.parameters.items()):
        if not supported_param(param):
            error_fn("We do not support positional-only args, varargs, or varkwargs.")

        if param.kind == inspect.Parameter.KEYWORD_ONLY:
            # The first time we see a kwarg-only arg, add "*" to the schema.
            if not saw_kwarg_only_arg:
                params.append("*")
                saw_kwarg_only_arg = True

        if param.annotation is inspect.Parameter.empty:
            error_fn(f"Parameter {name} must have a type annotation.")

        # The annotation might be converted to a string by annotation,
        # we convert it to the actual type.
        annotation_type, _ = unstringify_type(param.annotation)

        if annotation_type not in SUPPORTED_PARAM_TYPES:
            if annotation_type.__origin__ is tuple:
                list_type = tuple_to_list(annotation_type)
                example_type_str = "\n\n"
                # Only suggest the list type if this type is supported.
                if list_type in SUPPORTED_PARAM_TYPES.keys():
                    example_type_str = f"For example, {list_type}.\n\n"
                error_fn(
                    f"Parameter {name} has unsupported type {param.annotation}. "
                    f"We do not support Tuple inputs in schema. As a workaround, please try to use List instead. "
                    f"{example_type_str}"
                    f"The valid types are: {SUPPORTED_PARAM_TYPES.keys()}."
                )
            else:
                error_fn(
                    f"Parameter {name} has unsupported type {param.annotation}. "
                    f"The valid types are: {SUPPORTED_PARAM_TYPES.keys()}."
                )

        schema_type = SUPPORTED_PARAM_TYPES[annotation_type]
        if type(mutates_args) == str:
            if mutates_args != UNKNOWN_MUTATES:
                raise ValueError(
                    "mutates_args must either be a sequence of the names of "
                    "the arguments that are mutated or the string 'unknown'. "
                )
            if schema_type.startswith("Tensor"):
                schema_type = f"Tensor(a{idx}!){schema_type[len('Tensor'):]}"
        elif name in mutates_args:
            if not schema_type.startswith("Tensor"):
                error_fn(
                    f"Parameter {name} is in mutable_args but only Tensors or collections of Tensors can be mutated"
                )
            schema_type = f"Tensor(a{idx}!){schema_type[len('Tensor'):]}"
        seen_args.add(name)
        if param.default is inspect.Parameter.empty:
            params.append(f"{schema_type} {name}")
        else:
            default_repr = None
            if param.default is None or isinstance(param.default, (int, float, bool)):
                default_repr = str(param.default)
            elif isinstance(param.default, (str, torch.device)):
                default_repr = f'"{param.default}"'
            elif isinstance(param.default, torch.dtype):
                dtype_repr = str(param.default)
                torch_dot = "torch."
                assert dtype_repr.startswith(torch_dot)
                default_repr = dtype_repr[len(torch_dot) :]
            else:
                error_fn(
                    f"Parameter {name} has an unsupported default value type {type(param.default)}. "
                    f"Please file an issue on GitHub so we can prioritize this."
                )
            params.append(f"{schema_type} {name}={default_repr}")
    if mutates_args != UNKNOWN_MUTATES:
        mutates_args_not_seen = set(mutates_args) - seen_args
        if len(mutates_args_not_seen) > 0:
            error_fn(
                f"{mutates_args_not_seen} in mutates_args were not found in "
                f"the custom op's signature. "
                f"mutates_args should contain the names of all args that the "
                f"custom op mutates, or just the string 'unknown' if you don't know."
            )
    return_annotation, _ = unstringify_type(sig.return_annotation)
    ret = parse_return(return_annotation, error_fn)
    if op_name is not None:
        return f"{op_name}({', '.join(params)}) -> {ret}"
    return f"({', '.join(params)}) -> {ret}"


def derived_types(
    base_type: Union[type, typing._SpecialForm],
    cpp_type: str,
    list_base: bool,
    optional_base_list: bool,
    optional_list_base: bool,
):
    result: list[tuple[Union[type, typing._SpecialForm, GenericAlias], str]] = [
        (base_type, cpp_type),
        (typing.Optional[base_type], f"{cpp_type}?"),
    ]

    def derived_seq_types(typ: Union[type, typing._SpecialForm]):
        return (
            typing.Sequence[typ],  # type: ignore[valid-type]  # noqa: UP006
            typing.List[typ],  # type: ignore[valid-type]  # noqa: UP006
            GenericAlias(collections.abc.Sequence, (typ,)),
            GenericAlias(list, (typ,)),
        )

    if list_base:
        result.extend(
            (seq_typ, f"{cpp_type}[]") for seq_typ in derived_seq_types(base_type)
        )
    if optional_base_list:
        result.extend(
            (seq_typ, f"{cpp_type}?[]")
            for seq_typ in derived_seq_types(typing.Optional[base_type])
        )
    if optional_list_base:
        result.extend(
            (typing.Optional[seq_typ], f"{cpp_type}[]?")
            for seq_typ in derived_seq_types(base_type)
        )
    return result


def get_supported_param_types():
    data: list[tuple[Union[type, typing._SpecialForm], str, bool, bool, bool]] = [
        # (python type, schema type, type[] variant, type?[] variant, type[]? variant
        (Tensor, "Tensor", True, True, False),
        (int, "SymInt", True, False, True),
        (float, "float", True, False, True),
        (bool, "bool", True, False, True),
        (str, "str", False, False, False),
        (types.Number, "Scalar", True, False, False),
        (dtype, "ScalarType", False, False, False),
        (device, "Device", False, False, False),
    ]
    result = []
    for line in data:
        result.extend(derived_types(*line))
    return dict(result)


SUPPORTED_RETURN_TYPES = {
    Tensor: "Tensor",
    typing.List[Tensor]: "Tensor[]",  # noqa: UP006
    list[Tensor]: "Tensor[]",
    int: "SymInt",
    float: "float",
    bool: "bool",
    types.Number: "Scalar",
}


def parse_return(annotation, error_fn):
    if annotation is None:
        return "()"

    if annotation is inspect.Parameter.empty:
        error_fn("No return type annotation was provided. Please add one.")

    origin = typing.get_origin(annotation)
    if origin is not tuple:
        if annotation not in SUPPORTED_RETURN_TYPES.keys():
            error_fn(
                f"Return has unsupported type {annotation}. "
                f"The valid types are: {SUPPORTED_RETURN_TYPES}."
            )
        return SUPPORTED_RETURN_TYPES[annotation]

    args = typing.get_args(annotation)
    for arg in args:
        if arg not in SUPPORTED_RETURN_TYPES:
            error_fn(
                f"Return has unsupported type {annotation}. "
                f"The valid types are: {SUPPORTED_RETURN_TYPES}."
            )

    return "(" + ", ".join([SUPPORTED_RETURN_TYPES[arg] for arg in args]) + ")"


SUPPORTED_PARAM_TYPES = get_supported_param_types()


def supported_param(param: inspect.Parameter) -> bool:
    return param.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


def tuple_to_list(tuple_type: type[tuple]) -> type[list]:
    """
    Convert `tuple_type` into a list type with the same type arguments. Assumes that `tuple_type` is typing.Tuple type.
    """
    type_args = getattr(tuple_type, "__args__", None)
    # Account for different python versions, e.g. python 3.8 would give ()
    # but python 3.12 would give None.
    if (
        tuple_type is typing.Tuple  # noqa: UP006
        or tuple_type is tuple
        or type_args == ()
        or type_args is None
    ):
        # Handle the case of an empty tuple type
        return list
    elif len(type_args) == 1:
        # General case: create a List with the same type arguments
        return list[type_args[0]]  # type: ignore[valid-type]
    elif len(type_args) == 2 and type_args[1] is Ellipsis:
        return list[type_args[0]]  # type: ignore[valid-type]
    else:
        return list[typing.Union[tuple(type_args)]]  # type: ignore[misc, return-value]


# Below is an implementation of generating FunctionSchema from example values.
# This is helpful for generating FunctionSchema for HigherOrderOperator, where
# we don't have a function to inspect and each call of the higher order operator
# would have different schema.
@dataclass(frozen=True)
class HopArgumentInfo:
    # Could give a name to the operand by default it's empty string.
    name: str
    example_value: Any
    # Provide an default_value
    default_value: Any
    # Whether this arugment gets mutated in the hop subgraph.
    # For output, this should always be False
    is_mutated: bool


class HopArgumentInfoGen:
    @staticmethod
    def from_example(
        example_value: Any,
        *,
        name: str = "",
        default_value: Optional[Any],
        is_mutated: bool = False,
    ) -> HopArgumentInfo:
        if default_value is not None:
            assert type(example_value) == type(default_value)
        return HopArgumentInfo(
            name=name,
            example_value=example_value,
            default_value=default_value,
            is_mutated=is_mutated,
        )


class CTypeGen:
    convert_to_base_ty = {
        int: torch._C.IntType.get(),
        float: torch._C.FloatType.get(),
        str: torch._C.StringType.get(),
        bool: torch._C.BoolType.get(),
    }

    # should return torch._C.JitType but that annotation is busted
    @staticmethod
    def from_example(obj: Any) -> Any:
        import torch

        if isinstance(obj, torch.Tensor):
            return torch._C.TensorType.get()
        elif isinstance(obj, torch.fx.GraphModule):
            return torch._C.AnyType.get()
        elif isinstance(obj, torch.SymInt):
            return (torch._C.SymIntType.get(),)
        elif isinstance(obj, torch.SymBool):
            return (torch._C.SymBoolType.get(),)
        elif isinstance(obj, tuple):
            return torch._C._create_tuple_type(
                tuple(CTypeGen.from_example(arg) for arg in obj)
            )
        elif obj is None:
            return torch._C.NoneType.get()
        tp = type(obj)
        if tp not in CTypeGen.convert_to_base_ty:
            raise RuntimeError(f"unsupported type {tp}")
        return CTypeGen.convert_to_base_ty[tp]


class CArgumentGen:
    @staticmethod
    def from_hop_argument_info(arg_idx: int, arg_info: HopArgumentInfo) -> Any:
        return torch._C._create_argument(
            arg_info.name,
            CTypeGen.from_example(arg_info.example_value),
            None,
            arg_info.default_value,
            False,
            torch._C._create_alias_info(f"a{arg_idx}", arg_info.is_mutated),
        )


class CFunctionSchemaGen:
    """
    Note: [HigherOrderOperator schema generation]
    Each invocation of a HigherOrderOperator will have a different schema.
    For example, the schema of torch.cond varies depending on the true_fn and
    false_fn. So we need a way to generate the schema for each invocation of a HOP.

    We want to enforce the following invariants for HOP's schema:
        1. Flattened inputs. There should be no pytree structure in it.
        2. Flattened outputs. Note even if the hop returns a single value, it should be wrapped as a tuple.
        3. No aliasing. This includes inp-inp aliasing, inp-out aliasing and out-out aliasing.

    By enforcing these invariants, we could make HOP's schema meets the requirement of schema parser
    and makes hop easier to handle downstream. For example, suppose we have an invoke_quant_test HOP:

    class GraphModule(torch.nn.Module):
        def forward(self, l_x_, l_y_):
            subgraph_0 = self.subgraph_0
            invoke_quant_test = torch.ops.higher_order.invoke_quant_test(subgraph_0, l_x_, l_y_, scheme = 'nf4');

        class subgraph_0(torch.nn.Module):
            def forward(self, l_x_, l_y_):
                add_ = l_x_.add_(1)
                matmul = l_x_ @ l_y_
                sin = matmul.sin()
                child = sin.cos()
                child_1 = l_x_ + l_y_
                child_2 = l_x_ - l_y_
                child_3 = l_x_ @ l_y_
                return (child, child_1, child_2, child_3)

    By encoding the inputs of hop into a list of HopArgumentInfo and output as a single HopArgumentInfo,
    we would get the following schema:
        invoke_quant_test(Any arg0, Tensor(!) arg1, Tensor arg2, str scheme="\\"nf4\\"") -> ((Tensor, Tensor, Tensor, Tensor) out)
    """

    @staticmethod
    def from_hop_argument_info(
        op_name: str,
        inp_argument_info: list[HopArgumentInfo],
        out_argument_info: HopArgumentInfo,
    ) -> Any:
        args = []
        for i, arg_info in enumerate(inp_argument_info):
            args.append(CArgumentGen.from_hop_argument_info(i, arg_info))

        # NOTE: we want the output to always be a single argument with torch._C.TupleType.
        assert isinstance(
            out_argument_info.example_value, tuple
        ), f"expect out_argument_info's example_value to be a tuple but got {out_argument_info.example_value}"
        assert (
            not out_argument_info.is_mutated
        ), "out_argument_info.is_mutated should always be set to False."
        ret = CArgumentGen.from_hop_argument_info(0, out_argument_info)

        return torch._C._create_function_schema(
            op_name,
            "",
            args,
            [ret],
            False,
            False,
        )
