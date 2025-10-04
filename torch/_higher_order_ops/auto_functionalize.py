# mypy: allow-untyped-defs
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, get_args, Optional, Union

import torch
import torch._library.utils as library_utils
import torch.utils._pytree as pytree
from torch import Tensor
from torch._C import DispatchKey
from torch._higher_order_ops.utils import (
    _has_gen_schema,
    call_op,
    HopInstance,
    HopSchema,
    materialize_callable_in_args,
    unique_graph_id,
)
from torch._ops import HigherOrderOperator, OperatorBase, OpOverload
from torch._prims_common import clone_preserve_strides
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)


class SchemaHolder:
    def __init__(self, schema: torch.FunctionSchema):
        self.schema = schema

    def __eq__(self, other):
        return self.schema == other.schema

    def __hash__(self) -> int:
        return hash(self.schema)

    @classmethod
    def from_tree_spec(cls, tree_spec: pytree.TreeSpec):
        assert tree_spec is not None
        return cls(pytree.tree_unflatten([], tree_spec).schema)


# regsiter_constant allows us to get a tree_spec from pytree.tree_flatten(SchemaHolder(FunctionSchema)).
# The tree_spec is proxable in the graph and we can get back the schema via
# schema = pytree.tree_unflatten([], tree_spec).schema
pytree.register_constant(SchemaHolder)


def get_base(tensor):
    if torch.is_inference_mode_enabled():
        return tensor._inference_mode_base
    else:
        return tensor._base


class ViewInfo(ABC):
    base_index: int

    def __init__(self, base_index):
        self.base_index = base_index

    @abstractmethod
    def regenerate_view(self, bases_list: list[Tensor]):
        pass


@dataclass
class AsStridedViewInfo(ViewInfo):
    size: Sequence[Union[int, torch.SymInt]]
    stride: Sequence[Union[int, torch.SymInt]]
    storage_offset: int

    def __init__(self, base_index, size, stride, storage_offset):
        super().__init__(base_index)
        self.size = size
        self.stride = stride
        self.storage_offset = storage_offset

    def regenerate_view(self, bases_list: list[Tensor]):
        return torch.as_strided(
            bases_list[self.base_index],
            self.size,
            self.stride,
            self.storage_offset,
        )


@dataclass
class SliceViewInfo(ViewInfo):
    dim: Union[int, torch.SymInt]
    start: Union[int, torch.SymInt]
    end: Union[int, torch.SymInt]

    def __init__(self, base_index, dim, start, end):
        super().__init__(base_index)
        self.dim = dim
        self.start = start
        self.end = end

    def regenerate_view(self, bases_list: list[Tensor]):
        return torch.ops.aten.slice.Tensor(
            bases_list[self.base_index], self.dim, self.start, self.end
        )


@dataclass
class AliasViewInfo(ViewInfo):
    def __init__(self, base_index):
        super().__init__(base_index)

    def regenerate_view(self, bases_list: list[Tensor]):
        return torch.ops.aten.alias.default(bases_list[self.base_index])


@dataclass
class NotView(ViewInfo):
    def __init__(self, base_index):
        super().__init__(base_index)

    def regenerate_view(self, bases_list: list[Tensor]):
        return bases_list[self.base_index]


def is_alias(base, tensor):
    from torch.fx.experimental.symbolic_shapes import statically_known_true, sym_eq

    return all(
        statically_known_true(a)
        for a in [
            sym_eq(base.storage_offset(), tensor.storage_offset()),
            sym_eq(base.stride(), tensor.stride()),
            sym_eq(base.size(), tensor.size()),
        ]
    )


# return None or (dim, start, end)
def try_use_slice(base, tensor):
    from torch.fx.experimental.symbolic_shapes import statically_known_true, sym_eq

    # This condition should never be triggered.
    if is_alias(base, tensor):
        return (0, 0, base.size()[0])

    # TODO is there cases can we use slice even if stride or len(sizes) are not equal?
    if not statically_known_true(sym_eq(tensor.stride(), base.stride())):
        return None
    if not statically_known_true(sym_eq(len(tensor.size()), len(base.size()))):
        return None

    dim = None
    count = 0
    for i in range(len(tensor.size())):
        if base.size()[i] != tensor.size()[i]:
            dim = i
            count = count + 1
    if count != 1:
        return None

    if tensor.storage_offset() % tensor.stride()[dim] != 0:
        return None
    start = tensor.storage_offset() // tensor.stride()[dim]
    end = start + tensor.size()[dim]
    return (dim, start, end)


def write_view_information_to_args(
    mutable_arg_names: list[str],
    mutable_arg_types: list[torch.Type],
    kwargs: dict[str, Any],
    arg_to_base_index: dict[str, Any],
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

        assert f"{prefix}_slice_dim" not in kwargs
        assert f"{prefix}_slice_start" not in kwargs
        assert f"{prefix}_slice_end" not in kwargs

        def use_as_strided(tensor):
            kwargs[f"{prefix}_size"] = tensor.size()
            kwargs[f"{prefix}_stride"] = tensor.stride()
            kwargs[f"{prefix}_storage_offset"] = tensor.storage_offset()

        def use_slice(dim, start, end):
            kwargs[f"{prefix}_slice_dim"] = dim
            kwargs[f"{prefix}_slice_start"] = start
            kwargs[f"{prefix}_slice_end"] = end

        def use_alias():
            kwargs[f"{prefix}_alias"] = True

        # The start if the function
        if tensor is None:
            kwargs[f"{prefix}_base_index"] = None
        else:
            base = get_base(tensor)
            kwargs[f"{prefix}_base_index"] = base_index
            if base is None:
                # no need to add anything else other than _base_index
                return
            elif is_alias(base, tensor):
                use_alias()
            elif (slice_info := try_use_slice(base, tensor)) is not None:
                use_slice(*slice_info)
            else:
                use_as_strided(tensor)

    for arg_name, arg_type in zip(mutable_arg_names, mutable_arg_types):
        arg = kwargs[arg_name]
        if library_utils.is_tensorlist_like_type(arg_type):
            if arg is None:
                kwargs[f"_{arg_name}_length"] = None
            else:
                kwargs[f"_{arg_name}_length"] = len(arg)
                for i, elem in enumerate(arg):
                    write_single_view(
                        f"_{arg_name}_{i}", elem, arg_to_base_index[arg_name][i]
                    )

        elif library_utils.is_tensor_like_type(arg_type):
            write_single_view(
                f"_{arg_name}",
                kwargs[arg_name],
                arg_to_base_index.get(arg_name, None),  # type: ignore[arg-type]
            )
        else:
            raise RuntimeError(f"Unsupported type {arg_type}")


# Returns a dict of arg_name -> ViewInfo | [ViewInfo]
def read_view_information_from_args(
    mutable_arg_names: list[str],
    mutable_arg_types: list[torch.Type],
    kwargs: dict[str, Any],
    all_bases: list[Tensor],
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
        elif f"{prefix}_alias" in kwargs:
            get_arg(f"{prefix}_alias")
            return AliasViewInfo(base_index)
        elif f"{prefix}_storage_offset" in kwargs:
            # The view is regenerated using as_strided.
            size = get_arg(f"{prefix}_size")
            stride = get_arg(f"{prefix}_stride")
            storage_offset = get_arg(f"{prefix}_storage_offset")
            return AsStridedViewInfo(base_index, size, stride, storage_offset)
        elif f"{prefix}_slice_dim" in kwargs:
            dim = get_arg(f"{prefix}_slice_dim")
            start = get_arg(f"{prefix}_slice_start")
            end = get_arg(f"{prefix}_slice_end")
            return SliceViewInfo(base_index, dim, start, end)
        else:
            # This means that the argument is the base tensor
            return NotView(base_index)

    args_view_info: dict[str, Any] = {}
    for arg_name, arg_type in zip(mutable_arg_names, mutable_arg_types):
        if library_utils.is_tensorlist_like_type(arg_type):
            length = get_arg(f"_{arg_name}_length")
            if length is None:
                # The whole list is None.
                args_view_info[arg_name] = None
            else:
                args_view_info[arg_name] = [
                    read_single_view(f"_{arg_name}_{i}") for i in range(length)
                ]

        elif library_utils.is_tensor_like_type(arg_type):
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
        super().__init__("auto_functionalized", cacheable=True)

    def __call__(
        self,
        /,
        _mutable_op: OpOverload,
        **kwargs: Any,
    ) -> tuple[Any, tuple[Tensor, ...]]:
        assert can_auto_functionalize(_mutable_op)
        assert isinstance(kwargs, dict)
        return super().__call__(_mutable_op, **kwargs)


auto_functionalized = AutoFunctionalized()
auto_functionalized.__module__ = "torch.ops.higher_order"

auto_functionalized.fallthrough(DispatchKey.AutogradCPU)
auto_functionalized.fallthrough(DispatchKey.AutogradCUDA)


_MutableOpType = Union[OpOverload, HigherOrderOperator]


class AutoFunctionalizedV2(HigherOrderOperator):
    """auto_functionalized_v2(_mutable_op, **kwargs)

    This HOP runs a "functional" version of _mutable_op.
    Unlike AutoFunctionalized, this version is improved to better handle
    view tensors. This version is only used in non export mode.
    """

    def __init__(self) -> None:
        super().__init__("auto_functionalized_v2", cacheable=True)

    def __call__(
        self,
        /,
        _mutable_op: _MutableOpType,
        **kwargs: Any,
    ) -> tuple[Any, tuple[Tensor, ...]]:
        _op_to_check: Optional[Union[OpOverload, HopInstance]] = None
        if isinstance(_mutable_op, HigherOrderOperator):
            _op_to_check = HopInstance(
                _mutable_op,
                SchemaHolder.from_tree_spec(kwargs.get("_op_schema", None)).schema,  # type: ignore[arg-type]
            )
        else:
            _op_to_check = _mutable_op

        assert _op_to_check is not None
        assert can_auto_functionalize(_op_to_check)
        assert isinstance(kwargs, dict)
        return super().__call__(_mutable_op, **kwargs)


auto_functionalized_v2 = AutoFunctionalizedV2()
auto_functionalized_v2.__module__ = "torch.ops.higher_order"

auto_functionalized_v2.fallthrough(DispatchKey.AutogradCPU)
auto_functionalized_v2.fallthrough(DispatchKey.AutogradCUDA)


def can_auto_functionalize(
    op: Union[OperatorBase, HopInstance],
) -> bool:
    if isinstance(op, HopInstance):
        # HOPs that implement gen_schema and schema is not functional are auto_functionalizable.
        if not _has_gen_schema(op._op):
            return False

    else:
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
        if torch._library.utils.is_tensor_like_type(arg.type):
            continue
        if torch._library.utils.is_tensorlist_like_type(arg.type):
            continue
        return False

    if len(schema.returns) == 1 and isinstance(schema.returns[0].type, torch.NoneType):
        # Skip schema returns -> None
        return True
    if isinstance(op, OpOverload):
        # The returns of OpOverload must not alias anything
        for ret in schema.returns:
            if ret.alias_info is None and type(ret.type) is torch.TensorType:
                continue
            # Not yet supported: List[Tensor] return.
            return False
        if torch._C._dispatch_has_kernel_for_dispatch_key(op.name(), "Functionalize"):
            return False
    return True


def get_mutable_args_from_schema(
    schema: torch.FunctionSchema,
) -> tuple[list[str], list[torch.Type]]:
    """
    Returns the list of argument names that get mutated according to the
    schema and their types.
    """
    mutable_args_names = [
        arg.name
        for arg in schema.arguments
        if arg.alias_info is not None and arg.alias_info.is_write
    ]

    mutable_args_types = [
        arg.type
        for arg in schema.arguments
        if arg.alias_info is not None and arg.alias_info.is_write
    ]
    return mutable_args_names, mutable_args_types  # type: ignore[return-value]


def get_mutable_args(op: OpOverload) -> tuple[list[str], list[torch.Type]]:
    return get_mutable_args_from_schema(op._schema)


def do_auto_functionalize(
    mode: "torch._subclasses.functional_tensor.FunctionalTensorMode",
    op: OpOverload,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Functionalizes a call to op(*args, **kwargs) by emitting a call to
    `outs = auto_functionalized(op, normalized_kwargs)`
    and replacing the mutated (args, kwargs) with the corresponding outputs.

    The normalized_kwargs are just the (args, kwargs), but all in kwarg form.
    This makes handling easier for the auto_functionalized HOP.
    """
    from torch._subclasses.functional_tensor import PythonFunctionalizeAPI

    ctx = PythonFunctionalizeAPI(mode=mode)

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
            # as it means the optional arg was passed with its default
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
            op,
            **unwrapped_kwargs,  # type: ignore[arg-type]
        )

    # List of the name of args that get mutated (according to the schema)
    mutable_args_names, _ = get_mutable_args(op)

    unwrapped_actual_out: Union[Any, tuple[Any]] = unwrapped_outs[
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


# Wrapper for GraphModule that applies functionalization during execution to enable
# epilogue graph inlining and better fusion opportunities in subgraphs
# When tracing this wrapper, we'll get a graph module with epilogue.
#
# We want to hash it according to the original graph module, so that when we go
# from Functional mode -> fake mode for multiple invoke_subgraph calls that share,
# the same inner graph module, we can hit the cache.
class FunctionalCallableWithEpilogue:
    def __init__(self, orig_callable: Callable):
        self.orig_callable = orig_callable

    def __call__(self, *args, **kwargs):
        # We call torch.func.functionalize. This allows us to inline the epilogue graph.
        # Inlining has the benefit of allowing easiser fusion inside subgraph.
        # Though the epilogue graph contains copy_, it is OK because inductor can handle it
        # and this is also how we have been supporting top-level graph input mutation.
        return tuple(torch.func.functionalize(self.orig_callable)(*args, **kwargs))

    def __hash__(self):
        return id(self.orig_callable)


def do_auto_functionalize_v2(
    mode: "torch._subclasses.functional_tensor.FunctionalTensorMode",
    op: Union[OpOverload, HopInstance],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    from torch._subclasses.functional_tensor import PythonFunctionalizeAPI

    ctx = PythonFunctionalizeAPI(mode=mode)

    # All of the (args, kwargs), but all as kwargs. The names for the
    # args come from the schema. This makes it easier for us to work with them.
    normalized_kwargs = {}

    schema = op._schema
    # pyrefly: ignore  # bad-assignment
    op = op._op if isinstance(op, HopInstance) else op
    assert isinstance(op, get_args(_MutableOpType))

    def _functionalize_callable(arg: Any):
        if callable(arg):
            return FunctionalCallableWithEpilogue(arg)
        return arg

    args, kwargs = pytree.tree_map(_functionalize_callable, (args, kwargs))

    for idx, arg in enumerate(schema.arguments):
        # NB: torch_dispatch kwargs are the args defined as kwarg-only in the schema
        if arg.name in kwargs:
            normalized_kwargs[arg.name] = kwargs[arg.name]
        elif idx < len(args):
            # if its out of bounds we don't need to do anything
            # as it means the optional arg was passed with its default
            # value
            normalized_kwargs[arg.name] = args[idx]
        else:
            normalized_kwargs[arg.name] = arg.default_value

    # List of the name of args that get mutated (according to the schema)
    mutable_args_names, mutable_args_types = get_mutable_args_from_schema(schema)

    # A list of all bases of mutable args without duplication
    all_bases = []
    all_bases_addresses: list[int] = []

    # Map arg_name to the index of its base in all_bases.
    arg_to_base_index: dict[str, Any] = {}

    def update_dict(tensor, arg_name, index=None):
        base = tensor if get_base(tensor) is None else get_base(tensor)

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

    assert "_all_bases" not in unwrapped_kwargs, (op, unwrapped_kwargs)
    auto_func_kwargs = dict(unwrapped_kwargs, _all_bases=all_basis_unwrapped)
    if isinstance(op, HigherOrderOperator):
        assert "_ops_schema" not in unwrapped_kwargs, (op, unwrapped_kwargs)
        # We pass in the tree_spec of tree_flatten(SchemaHolder) to make it proxable
        auto_func_kwargs.update(
            {"_op_schema": pytree.tree_flatten(SchemaHolder(schema))[1]}
        )

    with ctx.redispatch_to_next():
        unwrapped_outs = auto_functionalized_v2(
            op,
            **auto_func_kwargs,  # type: ignore[arg-type]
        )

    unwrapped_actual_out: Union[Any, tuple[Any]] = (
        unwrapped_outs if len(all_bases) == 0 else unwrapped_outs[: -len(all_bases)]
    )

    unwrapped_mutable_out = (
        [] if len(all_bases) == 0 else unwrapped_outs[-len(all_bases) :]
    )

    if isinstance(op, HigherOrderOperator):
        assert len(schema.returns) > 0, (
            f"hop is expected to return at least one output {schema}."
        )
        assert len(unwrapped_actual_out) == len(schema.returns)
    else:
        if len(schema.returns) == 0:
            assert unwrapped_actual_out[0] is None
            unwrapped_actual_out = None
        elif len(schema.returns) == 1:
            assert len(unwrapped_actual_out) == 1
            unwrapped_actual_out = unwrapped_actual_out[0]
        else:
            assert len(unwrapped_actual_out) == len(schema.returns)

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
    _only_clone_these_tensors: Optional[tuple[str, ...]] = None,
    **kwargs: Any,
) -> tuple[Any, tuple[Tensor, ...]]:
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
                else (
                    clone_preserve_strides(kwargs[name])
                    if kwargs[name] is not None
                    else None
                )
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
) -> tuple[Any, tuple[Tensor, ...]]:
    with mode:
        result = auto_functionalized_dense(
            _mutable_op, _only_clone_these_tensors=None, **kwargs
        )
        return result


@auto_functionalized.py_impl(ProxyTorchDispatchMode)
def auto_functionalized_proxy(
    mode,
    _mutable_op: OpOverload,
    **kwargs: Any,
) -> tuple[Any, tuple[Tensor, ...]]:
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
    _mutable_op: _MutableOpType,
    _only_clone_these_bases: Optional[tuple[int, ...]] = None,
    **kwargs: Any,
) -> tuple[Any, tuple[Tensor, ...]]:
    _all_bases: list[Tensor] = kwargs.pop("_all_bases", [])
    if _only_clone_these_bases is None:
        _only_clone_these_bases = tuple(range(len(_all_bases)))

    if isinstance(_mutable_op, OpOverload):
        schema: torch._C.FunctionSchema = _mutable_op._schema
    else:
        schema = pytree.tree_unflatten([], kwargs.pop("_op_schema")).schema

    if isinstance(_mutable_op, OpOverload):
        _callable_op: Union[HopInstance, OpOverload] = _mutable_op
    else:
        assert isinstance(schema, HopSchema)
        _callable_op = HopInstance(_mutable_op, schema)

    op_kwargs_new, all_bases_new = _generate_new_op_kwargs_from_bases(
        schema,
        kwargs,
        _all_bases,
        _only_clone_these_bases,
    )

    out = call_op(
        _callable_op,
        tuple(),
        op_kwargs_new,
    )

    if isinstance(out, tuple):
        return (*out, *all_bases_new)  # type: ignore[return-value]
    else:
        return (out, *all_bases_new)  # type: ignore[return-value]


def _generate_new_op_kwargs_from_bases(
    schema, kwargs, all_bases, _only_clone_these_bases
):
    mutable_args_names, mutable_args_types = get_mutable_args_from_schema(schema)
    args_view_info = read_view_information_from_args(
        mutable_args_names, mutable_args_types, kwargs, all_bases
    )

    def maybe_copy(i, t):
        if t is None:
            return None
        if i in _only_clone_these_bases:
            return clone_preserve_strides(t)
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

    return new_kwargs, all_bases_new


@auto_functionalized_v2.py_impl(FakeTensorMode)
def auto_functionalized_v2_fake(
    mode,
    _mutable_op: _MutableOpType,
    **kwargs: dict[str, Any],
) -> tuple[Any, tuple[Tensor, ...]]:
    with mode:
        result = auto_functionalized_v2_dense(
            _mutable_op, _only_clone_these_bases=None, **kwargs
        )
        return result


@auto_functionalized_v2.py_impl(ProxyTorchDispatchMode)
def auto_functionalized_v2_proxy(
    mode,
    _mutable_op: _MutableOpType,
    **kwargs: Any,
) -> tuple[Any, tuple[Tensor, ...]]:
    if isinstance(_mutable_op, HigherOrderOperator):
        # Note [materialize callable inputs as graph]
        # Below code materializes the callable inputs to the hop as graph modules.
        # kwargs may contain general callables, that are not proxable e.g. FunctionWithNoFreeVars
        # this could happen when we auto_functionalize the backward of the hop,
        # where backward fn is a callablle that wraps forward graph module.
        # This function materialize the callable args according to the schema of the hop.

        # We cannot materialize the callables in kwargs directly because the inputs to callable
        # vary from hops to hop. To make the materialiation process generic to all hops,
        # we trace a function that wraps the hop and let each hop itself figure out how to trace
        # its callable inputs. Then we look at the schema of the traced hop node and replace the
        # callable in original kwarg with the traced subgraphs.
        #
        # Specifically, we first trace a wrapped_fn that calls into the hop. Then we look for the
        # hop node in the traced graph and graph module inputs to the hop. Finally, we replace the
        # original kwarg's callable with the graph module.
        all_bases = kwargs.get("_all_bases", [])
        _only_clone_these_bases = kwargs.get("_only_clone_these_bases", None)
        if _only_clone_these_bases is None:
            _only_clone_these_bases = tuple(range(len(all_bases)))

        schema = pytree.tree_unflatten([], kwargs.get("_op_schema", None)).schema  # type: ignore[arg-type]
        new_kwargs, _ = _generate_new_op_kwargs_from_bases(
            schema,
            {k: v for k, v in kwargs.items() if k not in ("_all_bases", "_op_schema")},
            all_bases,
            _only_clone_these_bases,
        )

        _, materialized_kwargs = materialize_callable_in_args(
            HopInstance(_mutable_op, schema), tuple(), new_kwargs
        )

        # Only replace the callabes in kwargs with the materialized subgraphs.
        # The rest of the kwargs are kept unchanged.
        for k, v in kwargs.items():
            if callable(v):
                assert k in materialized_kwargs and isinstance(
                    materialized_kwargs[k], torch.fx.GraphModule
                )
                kwargs[k] = materialized_kwargs[k]

    with disable_proxy_modes_tracing():
        out = auto_functionalized_v2(_mutable_op, **kwargs)

    proxy_kwargs = pytree.tree_map(mode.tracer.unwrap_proxy, kwargs)

    if isinstance(_mutable_op, HigherOrderOperator):

        def _maybe_register_subgraph(val: Any):
            if isinstance(val, torch.fx.GraphModule):
                _, graph_name = unique_graph_id(
                    mode, prefix="auto_functionalized_subgraph"
                )
                mode.tracer.root.register_module(graph_name, val)
                return val
            return val

        proxy_kwargs = pytree.tree_map(_maybe_register_subgraph, proxy_kwargs)

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
