# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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


@dataclass
class ViewInfo:
    base_index: int
    size: Optional[Sequence[Union[int, torch.SymInt]]] = None
    stride: Optional[Sequence[Union[int, torch.SymInt]]] = None
    storage_offset: Optional[int] = None
    # When is_view is false, the tensor is the base, and
    # size, stride and storage_offset are all None.
    is_view: bool = True

    def regenerate_view(self, bases_list: List[Tensor]):
        if not self.is_view:
            return bases_list[self.base_index]

        assert self.stride is not None
        assert self.size is not None
        assert self.storage_offset is not None

        return torch.as_strided(
            bases_list[self.base_index],
            self.size,
            self.stride,
            self.storage_offset,
        )


def write_view_information_to_args(
    mutable_arg_names: List[str],
    mutable_arg_types: List[torch.Type],
    kwargs: Dict[str, Any],
    arg_to_base_index: Dict[str, Any],
):
    """
    This function writes the view information into kwargs. It reads mutable_args from kwargs.
    and uses arg_to_base_index and tensor information to write ViewInfo into kwargs.
    mutable_arg_names: mutable custom operator arg names.
    mutable_arg_types: mutable custom operator arg types.
    kwargs: the original custom operator args.
    arg_to_base_index: maps mutable_arg_name to int | [int] that refers to the base tensor that
                       corresponds to the input tensor
    """

    def write_single_view(prefix: str, tensor: Tensor, base_index: int):
        assert f"{prefix}_base_index" not in kwargs
        assert f"{prefix}_size" not in kwargs
        assert f"{prefix}_stride" not in kwargs
        assert f"{prefix}_storage_offset" not in kwargs

        if tensor is None:
            kwargs[f"{prefix}_base_index"] = None
        elif tensor._base is None:
            # if the tensor is the base (not view), for simplicity we do not serialize view meta.
            kwargs[f"{prefix}_base_index"] = base_index
        else:
            kwargs[f"{prefix}_base_index"] = base_index
            kwargs[f"{prefix}_size"] = tensor.size()
            kwargs[f"{prefix}_stride"] = tensor.stride()
            kwargs[f"{prefix}_storage_offset"] = tensor.storage_offset()

    for arg_name, arg_type in zip(mutable_arg_names, mutable_arg_types):
        arg = kwargs[arg_name]
        if isinstance(arg_type, torch.ListType):
            if arg is None:
                kwargs[f"_{arg_name}_length"] = None

            kwargs[f"_{arg_name}_length"] = len(arg)
            for i, elem in enumerate(arg):
                write_single_view(
                    f"_{arg_name}_{i}", elem, arg_to_base_index[arg_name][i]
                )

        elif isinstance(arg_type, (torch.TensorType, torch.OptionalType)):
            write_single_view(
                f"_{arg_name}",
                kwargs[arg_name],
                arg_to_base_index.get(arg_name, None),
            )
        else:
            raise RuntimeError(f"Unsupported type {arg_type}")


# Returns a dict of arg_name -> ViewInfo | [ViewInfo]
def read_view_information_from_args(
    mutable_arg_names: List[str],
    mutable_arg_types: List[torch.Type],
    kwargs: Dict[str, Any],
    all_bases: List[Tensor],
):
    """
    This reads the view information added by `write_view_information_to_args` from kwargs, pop them,
    and returns a dict arg_name -> ViewInfo | [ViewInfo](if the input is list). that maps each mutable arg
    to its view information.
    mutable_arg_names: mutable custom operator arg names.
    mutable_arg_types: mutable custom operator arg types.
    kwargs : args of auto_functionalize(custom_op, kwargs)
    """

    def get_arg(name):
        return kwargs.pop(name)

    def read_single_view(prefix):
        base_index = get_arg(f"{prefix}_base_index")
        if base_index is None:
            return None
        elif f"{prefix}_size" not in kwargs:
            assert f"{prefix}_stride" not in kwargs
            assert f"{prefix}_storage_offset" not in kwargs

            # This means that the argument is the base tensor
            return ViewInfo(base_index, all_bases[base_index], is_view=False)

        else:
            size = get_arg(f"{prefix}_size")
            stride = get_arg(f"{prefix}_stride")
            storage_offset = get_arg(f"{prefix}_storage_offset")
            return ViewInfo(base_index, size, stride, storage_offset, is_view=True)

    args_view_info: Dict[str, Any] = {}
    for arg_name, arg_type in zip(mutable_arg_names, mutable_arg_types):
        if isinstance(arg_type, torch.ListType):
            length = get_arg(f"_{arg_name}_length")
            if length is None:
                # The whole list is None.
                args_view_info[arg_name] = None
            else:
                args_view_info[arg_name] = [
                    read_single_view(f"_{arg_name}_{i}") for i in range(length)
                ]

        elif isinstance(arg_type, (torch.TensorType, torch.OptionalType)):
            args_view_info[arg_name] = read_single_view(f"_{arg_name}")
        else:
            raise RuntimeError(f"Unsupported type {arg_type}")
    return args_view_info


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
# auto_functionalize_v2 is an improved version of auto_functionalize that better handle
# re-inplacing views.


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

auto_functionalized.fallthrough(DispatchKey.AutogradCPU)
auto_functionalized.fallthrough(DispatchKey.AutogradCUDA)


class AutoFunctionalizedV2(HigherOrderOperator):
    """auto_functionalized_v2(_mutable_op, **kwargs)

    This HOP runs a "functional" version of _mutable_op.
    Unlike AutoFunctionalized, this version is improved to better handle
    view tensors. This version is only used in non export mode.
    """

    def __init__(self) -> None:
        super().__init__("auto_functionalized_v2")

    def __call__(
        self,
        /,
        _mutable_op: OpOverload,
        **kwargs: Any,
    ) -> Tuple[Any, Tuple[Tensor, ...]]:
        assert can_auto_functionalize(_mutable_op)
        assert isinstance(kwargs, dict)
        return super().__call__(_mutable_op, **kwargs)


auto_functionalized_v2 = AutoFunctionalizedV2()
auto_functionalized_v2.__module__ = "torch.ops.higher_order"

auto_functionalized_v2.fallthrough(DispatchKey.AutogradCPU)
auto_functionalized_v2.fallthrough(DispatchKey.AutogradCUDA)


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
    with ctx.redispatch_to_next():
        unwrapped_outs = auto_functionalized(
            op, **unwrapped_kwargs  # type: ignore[arg-type]
        )

    # List of the name of args that get mutated (according to the schema)
    mutable_args_names, _ = get_mutable_args(op)

    unwrapped_actual_out: Union[Any, Tuple[Any]] = unwrapped_outs[
        : -len(mutable_args_names)
    ]
    unwrapped_mutable_out = unwrapped_outs[-len(mutable_args_names) :]

    if len(op._schema.returns) == 0:
        assert unwrapped_actual_out[0] is None
        unwrapped_actual_out = None
    elif len(op._schema.returns) == 1:
        assert len(unwrapped_actual_out) == 1
        unwrapped_actual_out = unwrapped_actual_out[0]
    else:
        assert len(unwrapped_actual_out) == len(op._schema.returns)

    for name, unwrapped_out in zip(mutable_args_names, unwrapped_mutable_out):
        # Can be None if input was `Tensor(a!)?`
        if unwrapped_out is None:
            continue

        # We only handle Tensor or List[Tensor] here for now.
        def sync_update(o, orig_arg):
            ctx.replace(orig_arg, o)
            ctx.commit_update(orig_arg)
            ctx.sync(orig_arg)

        orig_arg = normalized_kwargs[name]

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


def do_auto_functionalize_v2(
    op: OpOverload,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Any:
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

    # List of the name of args that get mutated (according to the schema)
    mutable_args_names, mutable_args_types = get_mutable_args(op)

    # A list of all bases of mutable args without duplication
    all_bases = []
    all_bases_addresses: list[int] = []

    # Map arg_name to the index of its base in all_bases.
    arg_to_base_index: Dict[str, Any] = {}

    def update_dict(tensor, arg_name, index=None):
        base = tensor if tensor._base is None else tensor._base

        def set_result(base_index):
            if index is None:
                arg_to_base_index[arg_name] = base_index
            else:
                arg_to_base_index[arg_name][index] = base_index

        if not all_bases_addresses.__contains__(base._cdata):
            all_bases_addresses.append(base._cdata)
            all_bases.append(base)
            set_result(len(all_bases) - 1)
        else:
            set_result(all_bases_addresses.index(base._cdata))

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

                update_dict(tensor, arg_name, i)

        else:
            update_dict(arg, arg_name)

    # add view_meta for each args into unwrapped_kwargs.
    write_view_information_to_args(
        mutable_args_names,
        mutable_args_types,
        normalized_kwargs,
        arg_to_base_index,
    )

    # remove mutated args from the kwargs (its a function of _all_bases now)
    for arg_name in mutable_args_names:
        del normalized_kwargs[arg_name]  # type: ignore[arg-type]

    unwrapped_kwargs = ctx.unwrap_tensors(normalized_kwargs)  # type: ignore[arg-type]
    if "self" in unwrapped_kwargs or "self_" in unwrapped_kwargs:
        warnings.warn(
            "Using `self` or `self_` as an argument in the definition of custom ops may lead to ambiguous parsing. "
            "Please consider using a different name for this argument to avoid potential issues."
        )
    all_basis_unwrapped = ctx.unwrap_tensors(all_bases)

    with ctx.redispatch_to_next():
        unwrapped_outs = auto_functionalized_v2(
            op, **dict(unwrapped_kwargs, _all_bases=all_basis_unwrapped)  # type: ignore[arg-type]
        )

    unwrapped_actual_out: Union[Any, Tuple[Any]] = (
        unwrapped_outs if len(all_bases) == 0 else unwrapped_outs[: -len(all_bases)]
    )

    unwrapped_mutable_out = (
        [] if len(all_bases) == 0 else unwrapped_outs[-len(all_bases) :]
    )

    if len(op._schema.returns) == 0:
        assert unwrapped_actual_out[0] is None
        unwrapped_actual_out = None
    elif len(op._schema.returns) == 1:
        assert len(unwrapped_actual_out) == 1
        unwrapped_actual_out = unwrapped_actual_out[0]
    else:
        assert len(unwrapped_actual_out) == len(op._schema.returns)

    for orig_arg, unwrapped_out in zip(all_bases, unwrapped_mutable_out):
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


# auto_functionalize functions
@auto_functionalized.py_impl(DispatchKey.CompositeExplicitAutograd)
def auto_functionalized_dense(
    _mutable_op: OpOverload,
    _only_clone_these_tensors: Optional[Tuple[str, ...]] = None,
    **kwargs: Any,
) -> Tuple[Any, Tuple[Tensor, ...]]:
    new_kwargs = dict(**kwargs)
    result = []

    _mutable_args_names, _ = get_mutable_args(_mutable_op)
    for name in _mutable_args_names:
        if (
            _only_clone_these_tensors is not None
            and name not in _only_clone_these_tensors
        ):
            new_kwargs[name] = kwargs[name]
        else:
            new_kwargs[name] = (
                [clone_preserve_strides(x) for x in kwargs[name]]
                if kwargs[name] is not None and isinstance(kwargs[name], list)
                else clone_preserve_strides(kwargs[name])
                if kwargs[name] is not None
                else None
            )
        result.append(new_kwargs[name])
    out = _mutable_op(**new_kwargs)

    if isinstance(out, tuple):
        return (*out, *result)  # type: ignore[return-value]
    else:
        return (out, *result)  # type: ignore[return-value]


@auto_functionalized.py_impl(FakeTensorMode)
def auto_functionalized_fake(
    mode,
    _mutable_op: OpOverload,
    **kwargs: Any,
) -> Tuple[Any, Tuple[Tensor, ...]]:
    with mode:
        result = auto_functionalized_dense(_mutable_op, **kwargs)
        return result


@auto_functionalized.py_impl(ProxyTorchDispatchMode)
def auto_functionalized_proxy(
    mode,
    _mutable_op: OpOverload,
    **kwargs: Any,
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


@auto_functionalized.py_functionalize_impl
def auto_functionalized_func(ctx, _mutable_op, **kwargs):
    unwrapped_kwargs = ctx.unwrap_tensors(kwargs)
    with ctx.redispatch_to_next():
        result = auto_functionalized(_mutable_op, **unwrapped_kwargs)
    return ctx.wrap_tensors(result)


# auto_functionalized_v2 functions
@auto_functionalized_v2.py_impl(DispatchKey.CompositeExplicitAutograd)
def auto_functionalized_v2_dense(
    _mutable_op: OpOverload,
    _only_clone_these_bases: Optional[Tuple[int, ...]] = None,
    **kwargs: Any,
) -> Tuple[Any, Tuple[Tensor, ...]]:
    all_bases: List[Tensor] = kwargs.pop("_all_bases", [])
    mutable_args_names, mutable_args_types = get_mutable_args(_mutable_op)
    args_view_info = read_view_information_from_args(
        mutable_args_names, mutable_args_types, kwargs, all_bases
    )

    if _only_clone_these_bases is None:
        _only_clone_these_bases = tuple(range(len(all_bases)))

    def maybe_copy(i, t):
        if t is None:
            return None
        if i in _only_clone_these_bases:
            return t.clone()
        else:
            return t

    all_bases_new = [maybe_copy(i, t) for i, t in enumerate(all_bases)]

    # create new args
    new_kwargs = dict(**kwargs)

    # re-generate all inputs from all_bases_new using args_view_info and add them to new_kwargs.
    for arg_name in mutable_args_names:
        if args_view_info[arg_name] is None:
            new_kwargs[arg_name] = None
        elif isinstance(args_view_info[arg_name], list):
            new_kwargs[arg_name] = []
            for i, elem in enumerate(args_view_info[arg_name]):
                if elem is None:
                    new_kwargs[arg_name].append(None)
                else:
                    view_info = args_view_info[arg_name][i]
                    new_kwargs[arg_name].append(
                        view_info.regenerate_view(all_bases_new)
                    )
        else:
            new_kwargs[arg_name] = args_view_info[arg_name].regenerate_view(
                all_bases_new
            )

    out = _mutable_op(**new_kwargs)

    if isinstance(out, tuple):
        return (*out, *all_bases_new)  # type: ignore[return-value]
    else:
        return (out, *all_bases_new)  # type: ignore[return-value]


@auto_functionalized_v2.py_impl(FakeTensorMode)
def auto_functionalized_v2_fake(
    mode,
    _mutable_op: OpOverload,
    **kwargs: Dict[str, Any],
) -> Tuple[Any, Tuple[Tensor, ...]]:
    with mode:
        result = auto_functionalized_v2_dense(_mutable_op, **kwargs)
        return result


@auto_functionalized_v2.py_impl(ProxyTorchDispatchMode)
def auto_functionalized_v2_proxy(
    mode,
    _mutable_op: OpOverload,
    **kwargs: Dict[str, Any],
) -> Tuple[Any, Tuple[Tensor, ...]]:
    with disable_proxy_modes_tracing():
        out = auto_functionalized_v2(_mutable_op, **kwargs)

    proxy_kwargs = pytree.tree_map(mode.tracer.unwrap_proxy, kwargs)
    out_proxy = mode.tracer.create_proxy(
        "call_function",
        auto_functionalized_v2,
        (_mutable_op,),
        proxy_kwargs,
    )
    result = track_tensor_tree(out, out_proxy, constant=None, tracer=mode.tracer)
    return result


@auto_functionalized_v2.py_functionalize_impl
def auto_functionalized_v2_func(ctx, _mutable_op, **kwargs):
    unwrapped_kwargs = ctx.unwrap_tensors(kwargs)
    with ctx.redispatch_to_next():
        result = auto_functionalized_v2(_mutable_op, **unwrapped_kwargs)
    return ctx.wrap_tensors(result)
