# mypy: allow-untyped-defs
import dataclasses
import inspect
import sys
from collections.abc import Callable, Iterable, Iterator
from typing import Any, Literal, overload

import torch
import torch.utils._pytree as pytree
import torchgen
from torch import _C, _utils_internal
from torch._ops import OpOverload


@dataclasses.dataclass
class Kernel:
    """Models a (function, source location)"""

    func: Callable
    source: str

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class RegistrationHandle:
    """Does something when someone calls .destroy() on it"""

    def __init__(self, on_destroy: Callable):
        self._on_destroy = on_destroy

    def destroy(self) -> None:
        self._on_destroy()


def get_source(stacklevel: int) -> str:
    """Get a string that represents the caller.

    Example: "/path/to/foo.py:42"

    Use stacklevel=1 to get the caller's source
    Use stacklevel=2 to get the caller's caller's source
    etc.
    """
    frame = inspect.getframeinfo(sys._getframe(stacklevel))
    source = f"{frame.filename}:{frame.lineno}"
    return source


def parse_namespace(qualname: str) -> tuple[str, str]:
    splits = qualname.split("::")
    if len(splits) != 2:
        raise ValueError(
            f"Expected `qualname` to be of the form "
            f'"namespace::name", but got {qualname}. '
            f"The qualname passed to the torch.library APIs must consist "
            f"of a namespace and a name, e.g. aten::sin"
        )
    return splits[0], splits[1]


def lookup_op(qualname: str) -> OpOverload:
    namespace, name = parse_namespace(qualname)
    if "." in name:
        name, overload = name.split(".")
    else:
        overload = "default"
    ns = getattr(torch.ops, namespace)
    packet = getattr(ns, name)
    return getattr(packet, overload)


def is_builtin(op: OpOverload) -> bool:
    if not isinstance(op, OpOverload):
        raise AssertionError(f"op must be OpOverload, got {type(op)}")
    return op.namespace in {"aten", "prim", "prims"}


def is_functional_schema(schema: Any, *, allow_valid_view: bool = False) -> bool:
    """Check if the schema is functional.

    An operator is functional if:
    - it does not mutate any of its inputs
    - If no view are allowed
        - it does not return a view on any of its inputs
    - If valid views are allowed
        - it is not a view or a view with a single input Tensor and single output Tensor
    - it has at least one return
    """

    def is_functional(schema):
        if schema.is_mutable:
            return False
        rets = schema.returns
        is_non_mutating_view = len(rets) > 0 and any(
            r.alias_info is not None and not r.alias_info.is_write for r in rets
        )
        num_tensor_inputs = 0
        num_tensor_outputs = 0

        if isinstance(schema, torch.FunctionSchema):
            for arg in schema.arguments:
                if isinstance(arg.type, torch.TensorType):
                    num_tensor_inputs += 1

            for ret in schema.returns:
                if isinstance(ret.type, torch.TensorType):
                    num_tensor_outputs += 1

        elif isinstance(schema, torchgen.model.FunctionSchema):
            for argument in schema.arguments.flat_non_out:
                if argument.type.is_tensor_like():
                    num_tensor_inputs += 1

            for ret_arg in schema.returns:
                if ret_arg.type.is_tensor_like():
                    num_tensor_outputs += 1

        if is_non_mutating_view:
            return allow_valid_view and (
                num_tensor_inputs == 1 and num_tensor_outputs == 1
            )
        if not schema.returns:
            return False
        return True

    if isinstance(schema, torch._C.FunctionSchema):
        return is_functional(schema)

    # Lazy import because not all PyTorch builds have torchgen
    from torchgen.model import FunctionSchema

    if isinstance(schema, str):
        schema = FunctionSchema.parse(schema)
    if not isinstance(schema, FunctionSchema):
        raise AssertionError(f"schema must be FunctionSchema, got {type(schema)}")
    return is_functional(schema)


# should be torch._C.JitType but that annotation is busted
def is_tensorlist_like_type(typ: Any) -> bool:
    return (
        typ == _C.ListType(_C.TensorType.get())
        or typ == _C.ListType(_C.OptionalType(_C.TensorType.get()))
        or typ == _C.OptionalType(_C.ListType(_C.TensorType.get()))
        or typ == _C.OptionalType(_C.ListType(_C.OptionalType(_C.TensorType.get())))
    )


# should be torch._C.JitType but that annotation is busted
def is_tensor_like_type(typ: Any) -> bool:
    return typ == _C.TensorType.get() or typ == _C.OptionalType(_C.TensorType.get())


def mutates_and_returns_first_arg(op: OpOverload):
    """Check if an op is an inplace aten op, i.e. it mutates and returns the first arg.

    TODO: torchgen/model.py's FunctionSchema.parse is the source of truth for this,
    but not all PyTorch builds have torchgen (due to the yaml dependency being weird).
    Figure this out.

    Example: add_(Tensor(a!) x, Tensor y) -> Tensor(a)
    """
    if op.namespace != "aten":
        return False
    schema = op._schema
    if len(schema.returns) != 1:
        return False
    if schema.returns[0].alias_info is None:
        return False
    alias_set = schema.returns[0].alias_info.after_set
    if len(alias_set) != 1:
        return False
    loc = next(iter(alias_set))
    if len(schema.arguments) < 1:
        return False
    first_arg = schema.arguments[0]
    if first_arg.alias_info is None:
        return False
    if not first_arg.alias_info.is_write:
        return False
    alias_set = first_arg.alias_info.after_set
    if len(alias_set) != 1:
        return False
    if loc != next(iter(alias_set)):
        return False
    for arg in schema.arguments[1:]:
        if arg.alias_info is not None:
            return False
    return True


def fill_defaults(schema, args, kwargs):
    new_args = []
    new_kwargs = {}
    for i in range(len(schema.arguments)):
        info = schema.arguments[i]
        if info.kwarg_only:
            if info.name in kwargs:
                new_kwargs[info.name] = kwargs[info.name]
            else:
                new_kwargs[info.name] = info.default_value
        else:
            if i < len(args):
                new_args.append(args[i])
            else:
                new_args.append(info.default_value)
    return tuple(new_args), new_kwargs


def zip_schema(
    schema: _C.FunctionSchema, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Iterable[tuple[_C.Argument, Any]]:
    """zips schema.arguments and (args, kwargs) together.

    Assumes that (args, kwargs) were the inputs to some torch._ops.OpOverload:
    that is, (args, kwargs) must be bindable to the schema (args, kwargs).
    """
    if len(schema.arguments) < len(args) + len(kwargs):
        raise AssertionError(
            f"schema has {len(schema.arguments)} arguments but got {len(args)} args and {len(kwargs)} kwargs"
        )
    for i in range(len(schema.arguments)):
        info = schema.arguments[i]
        if info.kwarg_only:
            if info.name in kwargs:
                yield info, kwargs[info.name]
            continue
        if i >= len(args):
            if not info.kwarg_only and info.name in kwargs:
                yield info, kwargs[info.name]
            # args that are equal to their default values are not populated
            # if they are followed by args that are equal to their defaults.
            # Skip these.
            continue
        yield info, args[i]
    return


def hop_schema_from_fx_node(node):
    from torchgen.gen_schema_utils import FunctionSchemaGen

    hop = node.target
    if not isinstance(hop, torch._ops.HigherOrderOperator):
        raise RuntimeError("fx_node's target must be a hop.")

    def _collect_example_val(node):
        meta_val = node.meta.get("val", None)
        if meta_val is None:
            if node.op != "get_attr":
                raise AssertionError(
                    f"node.op must be 'get_attr' when val is None, got {node.op!r}"
                )
            meta_val = getattr(node.graph.owning_module, node.target)
        return meta_val

    example_inputs = []
    for arg in node.args:
        if isinstance(arg, (torch.fx.Node, torch.fx.node.Node)):
            example_inputs.append(_collect_example_val(arg))
        elif isinstance(
            arg, (torch.fx.immutable_collections.immutable_list, list, tuple)
        ):
            example_inputs.append([_collect_example_val(x) for x in arg])
        else:
            raise RuntimeError(f"Unsupported arg type {type(arg)}")

    # Bound the arguments to make sure number of inputs are correct
    bound_args: inspect.BoundArguments = inspect.signature(hop.__call__).bind(
        *example_inputs
    )

    # We treat example_output as a single value in return. This is to differentiate 1. return a single val
    # vs 2. return a tuple with one element.
    example_output = _collect_example_val(node)
    return FunctionSchemaGen.from_example(
        hop._name, tuple(bound_args.arguments.items()), (list(example_output),)
    )


def can_generate_trivial_fake_impl(op: OpOverload) -> bool:
    if not isinstance(op, OpOverload):
        raise AssertionError(f"op must be OpOverload, got {type(op)}")
    if is_builtin(op):
        # We control the built-ins. These may (in rare cases)
        # do input metadata mutation (which we have banned on custom ops)
        return False
    schema = op._schema
    # It's suspicious if the op is not mutable but returns nothing, so we return False out of an abundance of caution
    if not schema.is_mutable:
        return False
    if len(schema.returns) > 0:
        return False
    # If the op returns nothing, then it has a trivial fake impl.
    return True


def requires_set_python_module() -> bool:
    """If an op was defined in C++ and extended from Python using the
    torch.library APIs, returns if we require that there have been a
    m.set_python_module("mylib.ops") call from C++ that associates
    the C++ op with a python module.
    """
    return getattr(_utils_internal, "REQUIRES_SET_PYTHON_MODULE", True)


def handle_dispatch_mode(curr_mode, op_overload, *args, **kwargs):
    if not isinstance(curr_mode, torch.utils._python_dispatch.TorchDispatchMode):
        raise AssertionError(
            f"curr_mode must be TorchDispatchMode, got {type(curr_mode)}"
        )
    args_flattened, _ = torch.utils._pytree.tree_flatten((args, kwargs.values()))
    # TODO: need to double check the semantics of the "types" argument to torch_dispatch.
    # It's generated in PyInterpreter.cpp, but seems to be generated in two places,
    # where in one case we only include tensors with the python key, and in another
    # we include **all** tensors.
    overload_types = [
        type(a)
        for a in args_flattened
        if isinstance(a, torch.Tensor)
        and torch._C._dispatch_keys(a).has(torch._C.DispatchKey.Python)
    ]
    # TODO: check that I got these args correct (in C++, we pass in "0000"??)

    return curr_mode.__torch_dispatch__(op_overload, overload_types, args, kwargs)


def has_kwarg_only_args(schema: _C.FunctionSchema):
    return any(a.kwarg_only for a in schema.arguments)


def has_kwarg_only_tensors(schema: _C.FunctionSchema):
    for a in schema.arguments:
        if not (is_tensor_like_type(a.type) or is_tensorlist_like_type(a.type)):
            continue
        if not a.kwarg_only:
            continue
        return True
    return False


def has_tensor_arg(schema: _C.FunctionSchema) -> bool:
    """
    Given a schema, returns True if the schema has a Tensor arg.
    A Tensor arg is any arg with a type annotation that might involve Tensor.
    """
    return any(
        (is_tensor_like_type(a.type) or is_tensorlist_like_type(a.type))
        for a in schema.arguments
    )


def get_device_arg_index(schema: _C.FunctionSchema) -> int | None:
    """
    Given a schema, returns the id of the `device: torch.device` argument.
    If it does not exist, returns None.
    """
    for index, arg in enumerate(schema.arguments):
        if arg.type is _C.DeviceObjType.get() and arg.name == "device":
            return index
    return None


def iter_tensors(
    args: tuple[Any], kwargs: dict[str, Any], allowed_nesting: int = 1
) -> Iterator[torch.Tensor]:
    def check(arg):
        if isinstance(arg, torch.Tensor):
            yield arg
        elif allowed_nesting > 0 and isinstance(arg, (tuple, list)):
            yield from iter_tensors(tuple(arg), {}, allowed_nesting - 1)

    for arg in args:
        yield from check(arg)
    for kwarg in kwargs.values():
        yield from check(kwarg)


def check_aliasing_constraint(name, prev, result, get_module=lambda: "???"):
    """
    custom operators' outputs must not alias any inputs or other outputs.
    """
    storages = {t.untyped_storage()._cdata for t in prev if isinstance(t, torch.Tensor)}
    tuple_result = result
    if not isinstance(result, tuple):
        tuple_result = (result,)
    for tensor in iter_tensors(tuple_result, {}):
        key = tensor.untyped_storage()._cdata
        if tensor.untyped_storage()._cdata in storages:
            raise RuntimeError(
                f"{name} (with implementation in {get_module()}): "
                f"The output of this custom operator (1) must not "
                f"also be an input to this custom operator and "
                f"(2) may not alias any inputs to this custom operator "
                f"or other returns. "
                f"The most common way to trigger this error is if "
                f"we have y = custom_op(x) and y and x are the same Tensor. "
                f"Please instead return a clone of the offending output "
                f"tensor(s) (e.g. return x.clone()) or refactor the custom "
                f"operator to not return y."
            )
        storages.add(key)


def _c_check_aliasing_constraint(name, args, kwargs, result, get_module=lambda: "???"):
    """
    custom operators' outputs must not have any aliases
    This version uses C++ implementation for perf.
    Only List container is supported.
    Tensors in Lists with not only Tensors are checked.
    """
    tuple_result = result
    if not isinstance(result, tuple):
        tuple_result = (result,)
    if _C._any_output_is_alias_to_input_or_output(args, kwargs, tuple_result):
        raise RuntimeError(
            f"{name} (with implementation in {get_module()}): "
            f"The output of this custom operator (1) must not "
            f"also be an input to this custom operator and "
            f"(2) may not alias any inputs to this custom operator "
            f"or other returns. "
            f"The most common way to trigger this error is if "
            f"we have y = custom_op(x) and y and x are the same Tensor. "
            f"Please instead return a clone of the offending output "
            f"tensor(s) (e.g. return x.clone()) or refactor the custom "
            f"operator to not return y."
        )


class MutationChecker:
    """
    Check if an operator mutated its arguments.
    Usage:

    checker = MutationChecker(op, flat_args, args_spec)
    op(*args, **kwargs)
    checker.check()
    """

    def __init__(self, op, flat_args, args_spec):
        self.op = op
        self.args_spec = args_spec
        self.flat_args = flat_args
        self.real_pre_hashes = [
            hash_tensor(a) if isinstance(a, torch.Tensor) else None for a in flat_args
        ]

    def check(self):
        real_post_hashes = [
            hash_tensor(a) if isinstance(a, torch.Tensor) else None
            for a in self.flat_args
        ]
        was_mutated = [
            not torch.equal(pre, post)
            and not (pre.isnan().all() and post.isnan().all())
            if isinstance(pre, torch.Tensor) and isinstance(post, torch.Tensor)
            else None
            for pre, post in zip(self.real_pre_hashes, real_post_hashes)
        ]
        was_mutated_args, was_mutated_kwargs = pytree.tree_unflatten(
            was_mutated, self.args_spec
        )
        for info, was_mutated in zip_schema(
            self.op._schema, was_mutated_args, was_mutated_kwargs
        ):

            def check_one(info, was_mutated):
                if info.is_write == was_mutated:
                    return
                raise RuntimeError(
                    f"{self.op._name}: for argument '{info.name}': the operator's schema "
                    f"{self.op._schema} specified that "
                    f"the operator {'mutates' if info.is_write else 'does not mutate'} "
                    f"the argument, but this seems to be empirically wrong. "
                    f"Please make the schema and operator behavior consistent. "
                    f"You can specify that an operator mutates a Tensor by "
                    f"e.g. changing its schema type from 'Tensor name' to 'Tensor(a!) name'"
                    f"(use different identifiers (a, b, c, ...) for different Tensors)"
                )

            if is_tensor_like_type(info.type):
                check_one(info, was_mutated)
            elif is_tensorlist_like_type(info.type):
                was_any_mutated = False if was_mutated is None else any(was_mutated)
                check_one(info, was_any_mutated)


def hash_tensor(t: torch.Tensor) -> torch.Tensor:
    """Some inexpensive hash. Used as a quick and dirty indicator for tensor mutation"""
    return t.detach().float().mean()


def has_fake_kernel(op: torch._ops.OpOverload) -> bool:
    """If an operator (that stays alive until FakeTensorMode) has a Fake kernel.
    Don't use this if the operator decomposes before FakeTensorMode.
    """
    if can_generate_trivial_fake_impl(op):
        return True
    name = op._name
    if torch._C._dispatch_has_kernel_for_dispatch_key(
        name, "CompositeImplicitAutograd"
    ):
        return True
    opdef = torch._library.custom_ops._maybe_get_opdef(name)
    if opdef is None:
        # the non-torch.library.custom_op path
        if torch._C._dispatch_has_kernel_for_dispatch_key(
            name, "CompositeExplicitAutograd"
        ):
            return True
        entry = torch._library.simple_registry.singleton.find(name)
        if entry.fake_impl.kernel is not None:
            return True
        if torch._C._dispatch_has_kernel_for_dispatch_key(name, "Meta"):
            return True
    else:
        # the torch.library.custom_op path
        if opdef._abstract_fn is not None:
            return True
    return False


def mutated_args_kwargs(schema: _C.FunctionSchema) -> tuple[list[int], list[str]]:
    idxs = []
    keys = []
    for i, info in enumerate(schema.arguments):
        if info.alias_info is not None and info.alias_info.is_write:
            if info.kwarg_only:
                keys.append(info.name)
            else:
                idxs.append(i)
    return idxs, keys


tags_by_priority = [
    _C.Tag.needs_exact_strides,
    _C.Tag.needs_contiguous_strides,
    _C.Tag.needs_fixed_stride_order,
    _C.Tag.flexible_layout,
]


# Case 1: with_default=True (or omitted). Return type is guaranteed to be a Tag.
@overload
def get_layout_constraint_tag(
    fn: Any, *, with_default: Literal[True] = True
) -> _C.Tag: ...


# Case 2: with_default=False. Return type can be a Tag or None.
@overload
def get_layout_constraint_tag(
    fn: Any, *, with_default: Literal[False]
) -> _C.Tag | None: ...


def get_layout_constraint_tag(fn, *, with_default=True):
    for tag in tags_by_priority:
        if tag in fn.tags:
            return tag
    if with_default:
        if is_builtin(fn):
            return _C.Tag.flexible_layout
        import torch._functorch
        from torch._functorch import config

        return getattr(torch._C.Tag, config.custom_op_default_layout_constraint)
    return None


# List of random functions that should be treated as impure
_RANDOM_FUNCTIONS = {
    torch.rand,
    torch.randn,
    torch.randint,
    torch.randperm,
    torch.rand_like,
    torch.randn_like,
    torch.randint_like,
    torch.normal,
    torch.poisson,
    torch.bernoulli,
    torch.multinomial,
}


def is_impure(
    op: Callable,
    *,
    args: tuple[Any, ...] | None = None,
    kwargs: dict[str, Any] | None = None,
    impure_random: bool = True,
) -> bool:
    """
    An operator is impure if it:
    - Mutates its inputs (has a mutable schema)
    - Has nondeterministic/random behavior that mutates RNG state
    - Is explicitly marked as effectful via torch.library._register_effectful_op

    Args:
        op: The operator to check (function, OpOverload, HigherOrderOperator, etc.)
        args: Optional arguments that would be passed to the callable
        kwargs: Optional keyword arguments that would be passed to the callable
        impure_random: Whether to treat random operations as impure (default: True)

    Returns:
        bool: True if the callable has side effects, False otherwise
    """
    # Import here to avoid circular dependencies
    from torch._higher_order_ops.effects import _get_effect
    from torch.fx.node import _side_effectful_functions

    if isinstance(op, torch._ops.OpOverload):
        schema = getattr(op, "_schema", None)
        if schema is not None and schema.is_mutable:
            return True

        if op in _side_effectful_functions:
            return True

        if _get_effect(op) is not None:
            return True

    if isinstance(op, torch._ops.HigherOrderOperator):
        if op in (
            torch.ops.higher_order.auto_functionalized,
            torch.ops.higher_order.auto_functionalized_v2,
        ):
            # Check if the auto-functionalized operator (the first argument) is
            # side-effectful
            if args and len(args) > 0:
                return args[0] in _side_effectful_functions

        if _get_effect(op) is not None:
            return True

        return False

    # Impure since it mutates RNG state
    if impure_random and getattr(op, "_nondeterministic_seeded", False):
        return True

    # Handle Python random functions that don't have _nondeterministic_seeded
    # but still affect global RNG state (issue #151524)
    # These should be impure regardless of impure_random setting to maintain
    # consistency between eager and compiled execution
    # All random operations are impure to ensure consistent behavior
    # between eager and compiled execution, regardless of generator usage
    if op in _RANDOM_FUNCTIONS:
        return True

    schema = getattr(op, "_schema", None)
    if schema is not None and schema.is_mutable:
        return True

    if op in _side_effectful_functions:
        return True

    return False
