# mypy: allow-untyped-defs
import dataclasses
import inspect
import sys
import warnings
from typing import Any, Callable, Dict, Iterable, Iterator, List, Tuple, Union

import torch
import torch.utils._pytree as pytree
from torch import _C, _utils_internal
from torch._ops import OpOverload


def warn_deploy(stacklevel=3):
    warnings.warn(
        "Python torch.library APIs do nothing under torch::deploy (multipy). "
        "Please instead use C++ custom operator registration APIs.",
        RuntimeWarning,
        stacklevel=stacklevel,
    )


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


def parse_namespace(qualname: str) -> Tuple[str, str]:
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
    assert isinstance(op, OpOverload)
    return op.namespace in {"aten", "prim", "prims"}


def is_functional_schema(schema: Any) -> bool:
    """Check if the schema is functional.

    An operator is functional if:
    - it does not mutate any of its inputs
    - it does not return a view on any of its inputs
    - it has at least one return
    """

    def is_functional(schema):
        if schema.is_mutable:
            return False
        rets = schema.returns
        is_non_mutating_view = len(rets) > 0 and any(
            r.alias_info is not None and not r.alias_info.is_write for r in rets
        )
        if is_non_mutating_view:
            return False
        if not schema.returns:
            return False
        return True

    if isinstance(schema, torch._C.FunctionSchema):
        return is_functional(schema)

    # Lazy import because not all PyTorch builds have torchgen
    from torchgen.model import FunctionSchema

    if isinstance(schema, str):
        schema = FunctionSchema.parse(schema)
    assert isinstance(schema, FunctionSchema)
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
    if not len(schema.returns) == 1:
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
    schema: _C.FunctionSchema, args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> Iterable[Tuple[_C.Argument, Any]]:
    """zips schema.arguments and (args, kwargs) together.

    Assumes that (args, kwargs) were the inputs to some torch._ops.OpOverload:
    that is, (args, kwargs) must be bindable to the schema (args, kwargs).
    """
    assert len(schema.arguments) >= len(args) + len(kwargs)
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
            assert node.op == "get_attr"
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
    assert isinstance(op, OpOverload)
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
    assert isinstance(curr_mode, torch.utils._python_dispatch.TorchDispatchMode)
    overload_types = []
    args_flattened, _ = torch.utils._pytree.tree_flatten((args, kwargs.values()))
    for a in args_flattened:
        # TODO: need to double check the semantics of the "types" argument to torch_dispatch.
        # It's generated in PyInterpreter.cpp, but seems to be generated in two places,
        # where in one case we only include tensors with the python key, and in another
        # we include **all** tensors.
        if isinstance(a, torch.Tensor) and torch._C._dispatch_keys(a).has(
            torch._C.DispatchKey.Python
        ):
            overload_types.append(type(a))
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


def get_device_arg_index(schema: _C.FunctionSchema) -> Union[int, None]:
    """
    Given a schema, returns the id of the `device: torch.device` argument.
    If it does not exist, returns None.
    """
    for index, arg in enumerate(schema.arguments):
        if arg.type is _C.DeviceObjType.get() and arg.name == "device":
            return index
    return None


def iter_tensors(
    args: Tuple[Any], kwargs: Dict[str, Any], allowed_nesting: int = 1
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
    storages = {id(t.untyped_storage()) for t in prev if isinstance(t, torch.Tensor)}
    tuple_result = result
    if not isinstance(result, tuple):
        tuple_result = (result,)
    for tensor in iter_tensors(tuple_result, {}):
        key = id(tensor.untyped_storage())
        if id(tensor.untyped_storage()) in storages:
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
                    f"the argument, but this seems to be emperically wrong. "
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


def mutated_args_kwargs(schema: _C.FunctionSchema) -> Tuple[List[int], List[str]]:
    idxs = []
    keys = []
    for i, info in enumerate(schema.arguments):
        if info.alias_info is not None and info.alias_info.is_write:
            if info.kwarg_only:
                keys.append(info.name)
            else:
                idxs.append(i)
    return idxs, keys
