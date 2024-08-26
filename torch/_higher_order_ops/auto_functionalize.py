# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.utils._pytree as pytree
from torch import Tensor
from torch._C import DispatchKey
from torch._ops import HigherOrderOperator, OperatorBase, OpOverload
from torch._prims_common import clone_preserve_strides
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)


# NOTE: [auto-functionalizing custom ops]
# Users may wish to torch.compile custom ops that mutate their inputs.
# torch.compile will automatically support this op without anyone needing
# to provide a functionalization kernel for it. Here's how.
#
# Let's say we have a hypothetical mylib::sin_(Tensor(a!) x) -> ()
# op. First, when FakeTensor sees this op:
# - If the schema says it returns nothing, we can generate a trivial
#   FakeTensor rule for it (that returns nothing).
# - Otherwise, the user needs to provide a FakeTensor impl (fake impl)
#
# Next, when Python FunctionalTensor sees the op, it will functionalize
# it by emitting a call to an auto_functionalize(op, ["x"], {"x": ...})
# HOP and replacing the mutated inputs with corresponding outputs of this HOP.
# This HOP effectively runs the functional version of the op when
# called: it clones inputs that will be mutated, runs the op, and
# then returns (output, Tensors with the new values)
#
# if the passed inputs are views of another inputs, we return the changed
# based tensor and regenerate the future views from it.


class AutoFunctionalized(HigherOrderOperator):
    """auto_functionalized(_mutable_op, **kwargs)

    This HOP runs a "functional" version of _mutable_op.

    Concretely, it looks at all the arguments that are mutable through
    _mutable_op's operator schema, clones those kwargs, runs
    `out = _mutable_op(**kwargs)` with the cloned values, and then returns the
    operator output concatenated with the cloned values that were mutated.

    We have some restrictions on `_mutable_op`.
    See `can_auto_functionalize` for the restrictions. We can likely lift
    many of these if users request it.

    The reason why _mutable_op is prefixed with an
    underscore is to prevent collisions with kwarg names in **kwargs.
    """

    def __init__(self) -> None:
        super().__init__("auto_functionalized")

    def __call__(
        self,
        /,
        _mutable_op: OpOverload,
        **kwargs: Any,
    ) -> Tuple[Any, Tuple[Tensor, ...]]:
        assert can_auto_functionalize(_mutable_op)
        assert isinstance(kwargs, dict)
        return super().__call__(_mutable_op, **kwargs)


auto_functionalized = AutoFunctionalized()
auto_functionalized.__module__ = "torch.ops.higher_order"


def can_auto_functionalize(op: OperatorBase) -> bool:
    if not isinstance(op, OpOverload):
        return False

    if torch._library.utils.is_builtin(op):
        # We control the built-ins. These may (in rare cases)
        # do input metadata mutation (which we have banned on custom ops)
        return False
    schema = op._schema
    if not schema.is_mutable:
        return False
    schema = op._schema

    for arg in schema.arguments:
        if arg.alias_info is None:
            continue
        if not arg.alias_info.is_write:
            continue
        if type(arg.type) is torch.TensorType:
            continue
        if (
            type(arg.type) is torch.OptionalType
            and type(arg.type.getElementType()) is torch.TensorType
        ):
            continue
        if (
            type(arg.type) is torch.ListType
            and type(arg.type.getElementType()) is torch.TensorType
        ):
            continue
        # Not yet supported: other Tensor types. This includes things like
        # Tensor?[], Tensor[]?.
        return False

    if len(schema.returns) == 1 and isinstance(schema.returns[0].type, torch.NoneType):
        # Skip schema returns -> None
        return True
    # The returns must not alias anything
    for ret in schema.returns:
        if ret.alias_info is None and type(ret.type) is torch.TensorType:
            continue
        # Not yet supported: List[Tensor] return.
        return False
    if torch._C._dispatch_has_kernel_for_dispatch_key(op.name(), "Functionalize"):
        return False
    return True


@dataclass
class ViewInfo:
    base_index: int
    base: Any
    size: Any
    stride: Any
    storage_offset: Any


def serialize_views_meta(
    arg_names, arg_types, input_kwargs, output_kwargs, arg_to_base_index
):
    def serialize_single_view(prefix, tensor, base_index):
        if tensor is None:
            output_kwargs[f"_{prefix}_base_index"] = None
        else:
            output_kwargs[f"_{prefix}_base_index"] = base_index
            output_kwargs[f"_{prefix}_size"] = tensor.size()
            output_kwargs[f"_{prefix}_stride"] = tensor.stride()
            output_kwargs[f"_{prefix}_storage_offset"] = tensor.storage_offset()

    for arg_name, arg_type in zip(arg_names, arg_types):
        arg = input_kwargs[arg_name]
        if isinstance(arg_type, torch.ListType):
            if arg is None:
                output_kwargs[f"_{arg_name}_length"] = None

            output_kwargs[f"_{arg_name}_length"] = len(arg)
            for i, elem in enumerate(arg):
                serialize_single_view(
                    f"_{arg_name}_{i}", elem, arg_to_base_index[arg_name][i]
                )

        elif isinstance(arg_type, (torch.TensorType, torch.OptionalType)):
            serialize_single_view(
                f"_{arg_name}",
                input_kwargs[arg_name],
                arg_to_base_index.get(arg_name, None),
            )
        else:
            raise RuntimeError(f"Unsupported type {arg_type}")


# read the serialized size(), stride(), storage_offset() and return them in a dict that maps each args to its ViewInfo.
def deserialize_views_meta(arg_names, arg_types, input_kwargs, all_bases, pop_args):
    def get_arg(name):
        if pop_args:
            return input_kwargs.pop(name)

        return input_kwargs[name]

    def deserialize_single_view(prefix):
        base_index = get_arg(f"_{prefix}_base_index")
        if base_index is None:
            return None
        else:
            size = get_arg(f"_{prefix}_size")
            stride = get_arg(f"_{prefix}_stride")
            storage_offset = get_arg(f"_{prefix}_storage_offset")
            return ViewInfo(
                base_index, all_bases[base_index], size, stride, storage_offset
            )

    args_view_info: Dict[str, Any] = {}
    for arg_name, arg_type in zip(arg_names, arg_types):
        if isinstance(arg_type, torch.ListType):
            length = get_arg(f"_{arg_name}_length")
            if length is None:
                # The whole list is None.
                args_view_info[arg_name] = None
            else:
                args_view_info[arg_name] = [
                    deserialize_single_view(f"_{arg_name}_{i}") for i in range(length)
                ]

        elif isinstance(arg_type, (torch.TensorType, torch.OptionalType)):
            args_view_info[arg_name] = deserialize_single_view(f"_{arg_name}")
        else:
            raise RuntimeError(f"Unsupported type {arg_type}")
    return args_view_info


@auto_functionalized.py_impl(DispatchKey.CompositeExplicitAutograd)
def auto_functionalized_dense(
    _mutable_op: OpOverload,
    _only_clone_these_tensors: Optional[Tuple[str, ...]] = None,
    **kwargs: Any,
) -> Tuple[Any, Tuple[Tensor, ...]]:
    all_bases: List[Tensor] = kwargs.pop("_all_bases", [])
    mutable_args_names, mutable_args_types = get_mutable_args(_mutable_op)
    args_view_info = deserialize_views_meta(
        mutable_args_names, mutable_args_types, kwargs, all_bases, pop_args=True
    )

    def regenerate_view(ViewInfo):
        return torch.as_strided(
            ViewInfo.base, ViewInfo.size, ViewInfo.stride, ViewInfo.storage_offset
        )

    # Re-generate all inputs from _all_bases using args_view_info and add them to kwargs.
    for arg_name in mutable_args_names:
        if args_view_info[arg_name] is None:
            kwargs[arg_name] = None
        elif isinstance(args_view_info[arg_name], list):
            kwargs[arg_name] = []
            for i, elem in enumerate(args_view_info[arg_name]):
                if elem is None:
                    kwargs[arg_name].append(None)
                else:
                    view_info = args_view_info[arg_name][i]
                    kwargs[arg_name].append(regenerate_view(view_info))
        else:
            kwargs[arg_name] = regenerate_view(args_view_info[arg_name])

    # create new args
    new_kwargs = dict(**kwargs)
    result = []

    for name in mutable_args_names:
        if (
            _only_clone_these_tensors is not None
            and name not in _only_clone_these_tensors
        ):
            new_kwargs[name] = kwargs[name]
        else:
            if kwargs[name] is not None and isinstance(kwargs[name], list):
                new_kwargs[name] = [clone_preserve_strides(x) for x in kwargs[name]]

            elif kwargs[name] is not None:
                new_kwargs[name] = clone_preserve_strides(kwargs[name])

    out = _mutable_op(**new_kwargs)

    def observe_mutation(base, mutation_source):
        if (
            mutation_source.size() == base.size()
            and mutation_source.stride() == base.stride()
            and mutation_source.storage_offset() == base.storage_offset()
        ):
            return mutation_source

        return base.as_strided_scatter(
            mutation_source,
            mutation_source.size(),
            mutation_source.stride(),
            mutation_source.storage_offset(),
        )

    for i, base in enumerate(all_bases):
        # TODO add a test to make sure we handle arguments that are passed as null correctly
        if base is None:
            raise RuntimeError("base is None")
            result.append(None)

        base_with_effects = base
        for arg_name in mutable_args_names:
            arg = new_kwargs[arg_name]

            if args_view_info[arg_name] is None:
                continue

            if arg is None:
                continue

            if (
                _only_clone_these_tensors is not None
                and name not in _only_clone_these_tensors
            ):
                # if the argument is mutated in place, base would have already observed the effect.
                continue

            if isinstance(arg, list):
                for j, elem in enumerate(arg):
                    if args_view_info[arg_name][j] is None:
                        continue
                    # check `base` is a base for the this argument
                    if args_view_info[arg_name][j].base_index != i:
                        continue

                    mutation_source = new_kwargs[name][j]
                    base_with_effects = observe_mutation(
                        base_with_effects, mutation_source
                    )

            else:
                if args_view_info[arg_name].base_index != i:
                    continue

                mutation_source = new_kwargs[arg_name]
                base_with_effects = observe_mutation(base_with_effects, mutation_source)

        result.append(base_with_effects)

    if isinstance(out, tuple):
        return (*out, *result)  # type: ignore[return-value]
    else:
        return (out, *result)  # type: ignore[return-value]


@auto_functionalized.py_impl(FakeTensorMode)
def auto_functionalized_fake(
    mode,
    _mutable_op: OpOverload,
    **kwargs: Dict[str, Any],
) -> Tuple[Any, Tuple[Tensor, ...]]:
    with mode:
        result = auto_functionalized_dense(_mutable_op, **kwargs)
        return result


@auto_functionalized.py_impl(ProxyTorchDispatchMode)
def auto_functionalized_proxy(
    mode,
    _mutable_op: OpOverload,
    **kwargs: Dict[str, Any],
) -> Tuple[Any, Tuple[Tensor, ...]]:
    with disable_proxy_modes_tracing():
        out = auto_functionalized(_mutable_op, **kwargs)

    proxy_kwargs = pytree.tree_map(mode.tracer.unwrap_proxy, kwargs)
    out_proxy = mode.tracer.create_proxy(
        "call_function",
        auto_functionalized,
        (_mutable_op,),
        proxy_kwargs,
    )
    result = track_tensor_tree(out, out_proxy, constant=None, tracer=mode.tracer)
    return result


auto_functionalized.fallthrough(DispatchKey.AutogradCPU)
auto_functionalized.fallthrough(DispatchKey.AutogradCUDA)


def get_mutable_args(op: OpOverload) -> Tuple[List[str], List[torch.Type]]:
    """
    Returns the list of argument names that get mutated according to the
    schema and their types.
    """
    mutable_args_names = [
        arg.name
        for arg in op._schema.arguments
        if arg.alias_info is not None and arg.alias_info.is_write
    ]

    mutable_args_types = [
        arg.type
        for arg in op._schema.arguments
        if arg.alias_info is not None and arg.alias_info.is_write
    ]
    return mutable_args_names, mutable_args_types


def do_auto_functionalize(
    op: OpOverload,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Any:
    """Functionalizes a call to op(*args, **kwargs) by emitting a call to
    `outs = auto_functionalized(op, normalized_kwargs)`
    and replacing the mutated (args, kwargs) with the corresponding outputs.

    The normalized_kwargs are just the (args, kwargs), but all in kwarg form.
    This makes handling easier for the auto_functionalized HOP.
    """
    from torch._subclasses.functional_tensor import PythonFunctionalizeAPI

    ctx = PythonFunctionalizeAPI()

    # All of the (args, kwargs), but all as kwargs. The names for the
    # args come from the schema. This makes it easier for us to work with them.
    normalized_kwargs = {}

    schema = op._schema
    for idx, arg in enumerate(schema.arguments):
        # NB: torch_dispatch kwargs are the args defined as kwarg-only in the schema
        if arg.name in kwargs:
            normalized_kwargs[arg.name] = kwargs[arg.name]
        elif idx < len(args):
            # if its out of bounds we don't need to do anything
            # as it means the the optional arg was passed with its default
            # value
            normalized_kwargs[arg.name] = args[idx]
        else:
            normalized_kwargs[arg.name] = arg.default_value

    unwrapped_kwargs = ctx.unwrap_tensors(normalized_kwargs)  # type: ignore[arg-type]

    if "self" in unwrapped_kwargs or "self_" in unwrapped_kwargs:
        warnings.warn(
            "Using `self` or `self_` as an argument in the definition of custom ops may lead to ambiguous parsing. "
            "Please consider using a different name for this argument to avoid potential issues."
        )
    # List of the name of args that get mutated (according to the schema)
    mutable_args_names, mutable_args_types = get_mutable_args(op)

    # A list of all bases of mutable args without duplication
    all_basis = []
    all_basis_addresses: list[int] = []

    # Map arg_name to the index of its base in all_basis.
    arg_to_base_index: Dict[str, Any] = {}
    for arg_name in mutable_args_names:
        arg = normalized_kwargs[arg_name]
        if arg is None:
            continue

        if isinstance(arg, list):
            arg_to_base_index[arg_name] = {}
            for i, tensor in enumerate(arg):
                if tensor is None:
                    arg_to_base_index[arg_name].append(None)
                    continue

                base = tensor if tensor._base is None else tensor._base

                if not all_basis_addresses.__contains__(base._cdata):
                    all_basis_addresses.append(base._cdata)
                    all_basis.append(base)
                    arg_to_base_index[arg_name][i] = len(all_basis) - 1
                else:
                    arg_to_base_index[arg_name][i] = all_basis_addresses.index(
                        base._cdata
                    )

        else:
            base = arg if arg._base is None else arg._base

            if not all_basis_addresses.__contains__(base._cdata):
                all_basis_addresses.append(base._cdata)
                all_basis.append(base)
                arg_to_base_index[arg_name] = len(all_basis) - 1
            else:
                arg_to_base_index[arg_name] = all_basis_addresses.index(base._cdata)

    # add view_meta for each args into unwrapped_kwargs.
    serialize_views_meta(
        mutable_args_names,
        mutable_args_types,
        normalized_kwargs,
        unwrapped_kwargs,
        arg_to_base_index,
    )

    # remove mutated args from the kwargs (its a function of _all_bases now)
    for arg_name in mutable_args_names:
        del unwrapped_kwargs[arg_name]  # type: ignore[arg-type]

    all_basis_unwrapped = ctx.unwrap_tensors(all_basis)

    with ctx.redispatch_to_next():
        unwrapped_outs = auto_functionalized(
            op, **dict(unwrapped_kwargs, _all_bases=all_basis_unwrapped)  # type: ignore[arg-type]
        )

    unwrapped_actual_out: Union[Any, Tuple[Any]] = (
        unwrapped_outs if len(all_basis) == 0 else unwrapped_outs[: -len(all_basis)]
    )

    unwrapped_mutable_out = (
        [] if len(all_basis) == 0 else unwrapped_outs[-len(all_basis) :]
    )

    if len(op._schema.returns) == 0:
        assert unwrapped_actual_out[0] is None
        unwrapped_actual_out = None
    elif len(op._schema.returns) == 1:
        assert len(unwrapped_actual_out) == 1
        unwrapped_actual_out = unwrapped_actual_out[0]
    else:
        assert len(unwrapped_actual_out) == len(op._schema.returns)

    for orig_arg, unwrapped_out in zip(all_basis, unwrapped_mutable_out):
        # Can be None if input was `Tensor(a!)?`
        if unwrapped_out is None:
            continue

        # We only handle Tensor or List[Tensor] here for now.
        def sync_update(o, orig_arg):
            ctx.replace(orig_arg, o)
            ctx.commit_update(orig_arg)
            ctx.sync(orig_arg)

        if isinstance(unwrapped_out, torch.Tensor):
            sync_update(unwrapped_out, orig_arg)
        elif isinstance(unwrapped_out, list) and all(
            isinstance(o, torch.Tensor) for o in unwrapped_out
        ):
            assert len(orig_arg) == len(unwrapped_out)
            for orig_a, o in zip(orig_arg, unwrapped_out):
                sync_update(o, orig_a)
        else:
            raise RuntimeError(
                f"unsupported type for auto-functionalization: {unwrapped_out}"
            )

    return ctx.wrap_tensors(unwrapped_actual_out)  # type: ignore[arg-type]


@auto_functionalized.py_functionalize_impl
def auto_functionalized_func(ctx, _mutable_op, **kwargs):
    unwrapped_kwargs = ctx.unwrap_tensors(kwargs)
    with ctx.redispatch_to_next():
        result = auto_functionalized(_mutable_op, **unwrapped_kwargs)
    return ctx.wrap_tensors(result)
