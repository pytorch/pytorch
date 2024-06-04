# mypy: allow-untyped-defs
import inspect
import typing

from .. import device, dtype, Tensor, types


def infer_schema(prototype_function: typing.Callable, mutates_args=()) -> str:
    """Given a function with type hints, parses a schema.

    We make some assumptions to make our lives easier that correspond to how people
    write custom ops in real life:
    - none of the outputs alias any of the inputs or each other.
    - only the args listed in mutates_args are being mutated.

    Callers (e.g. the custom ops API) are responsible for checking these assumptions.
    """
    sig = inspect.signature(prototype_function)

    def error_fn(what):
        raise ValueError(
            f"infer_schema(func): {what} " f"Got func with signature {sig})"
        )

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

        if param.annotation not in SUPPORTED_PARAM_TYPES.keys():
            error_fn(
                f"Parameter {name} has unsupported type {param.annotation}. "
                f"The valid types are: {SUPPORTED_PARAM_TYPES.keys()}."
            )

        schema_type = SUPPORTED_PARAM_TYPES[param.annotation]
        if name in mutates_args:
            if not schema_type.startswith("Tensor"):
                error_fn(
                    f"Parameter {name} is in mutable_args but only Tensors or collections of Tensors can be mutated"
                )
            schema_type = f"Tensor(a{idx}!){schema_type[len('Tensor'):]}"
        seen_args.add(name)
        if param.default is inspect.Parameter.empty:
            params.append(f"{schema_type} {name}")
        else:
            if param.default is not None and not isinstance(
                param.default, (int, float, bool)
            ):
                error_fn(
                    f"Parameter {name} has an unsupported default value (we only support "
                    f"int, float, bool, None). Please file an issue on GitHub so we can "
                    f"prioritize this."
                )
            params.append(f"{schema_type} {name}={param.default}")
    mutates_args_not_seen = set(mutates_args) - seen_args
    if len(mutates_args_not_seen) > 0:
        error_fn(
            f"{mutates_args_not_seen} in mutates_args were not found in "
            f"the custom op's signature. "
            f"mutates_args should contain the names of all args that the "
            f"custom op mutates."
        )
    ret = parse_return(sig.return_annotation, error_fn)
    return f"({', '.join(params)}) -> {ret}"


def derived_types(
    base_type, cpp_type, list_base, optional_base_list, optional_list_base
):
    result = [
        (base_type, cpp_type),
        (typing.Optional[base_type], f"{cpp_type}?"),
    ]

    def derived_seq_types(typ):
        return [
            typing.Sequence[typ],  # type: ignore[valid-type]
            typing.List[typ],  # type: ignore[valid-type]
        ]

    if list_base:
        for seq_typ in derived_seq_types(base_type):
            result.append((seq_typ, f"{cpp_type}[]"))  # type: ignore[valid-type]
    if optional_base_list:
        for seq_typ in derived_seq_types(typing.Optional[base_type]):
            result.append((seq_typ, f"{cpp_type}?[]"))  # type: ignore[valid-type]
    if optional_list_base:
        for seq_typ in derived_seq_types(base_type):  # type: ignore[valid-type]
            result.append((typing.Optional[seq_typ], f"{cpp_type}[]?"))  # type: ignore[valid-type]
    return result


def get_supported_param_types():
    data = [
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
    typing.List[Tensor]: "Tensor[]",
    int: "SymInt",
    float: "float",
    bool: "bool",
    types.Number: "Scalar",
}


def parse_return(annotation, error_fn):
    if annotation is None:
        return "()"

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
