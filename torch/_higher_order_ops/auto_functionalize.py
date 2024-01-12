from typing import Any, Dict, List, Tuple

import torch
import torch.utils._pytree as pytree
from torch import Tensor
from torch._C import DispatchKey
from torch._ops import HigherOrderOperator
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
# - Otherwise, the user needs to provide a FakeTensor rule (abstract impl)
#
# Next, when Python FunctionalTensor sees the op, it will functionalize
# it by emitting a call to an auto_functionalize(op, ["x"], {"x": ...})
# HOP and replacing the mutated inputs with corresponding outputs of this HOP.
# This HOP effectively runs the functional version of the op when
# called: it clones inputs that will be mutated, runs the op, and
# then returns (output, Tensors with the new values)


class AutoFunctionalized(HigherOrderOperator):
    """auto_functionalized(op, mutated_args_names, kwargs)

    This HOP runs a "functional" version of op.

    Concretely, it clones kwargs that `op` mutates (specified by
    mutated_args_names), runs `out = op(**kwargs)` with the cloned values,
    and then returns (out, Tuple of the cloned values that were mutated).

    We have some restrictions on `op`.
    See `can_auto_functionalize` for the restrictions. We can likely lift
    many of these if users request it.
    """

    def __init__(self):
        super().__init__("auto_functionalized")

    def __call__(
        self,
        op: torch._ops.OpOverload,
        mutated_args_names: List[str],
        kwargs: Dict[str, Any],
    ) -> Tuple[Any, Tuple[Tensor, ...]]:
        assert can_auto_functionalize(op)
        assert isinstance(mutated_args_names, list)
        assert isinstance(kwargs, dict)
        return super().__call__(op, mutated_args_names, kwargs)


auto_functionalized = AutoFunctionalized()


def can_auto_functionalize(op: torch._ops.OperatorBase) -> bool:
    if not isinstance(op, torch._ops.OpOverload):
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
        # Not yet supported: other Tensor types. This includes things like
        # Tensor[], Tensor?[], Tensor[]?.
        return False

    # The returns must not alias anything
    for ret in schema.returns:
        if ret.alias_info is None and type(ret.type) is torch.TensorType:
            continue
        # Not yet supported: List[Tensor] return.
        return False
    return True


@auto_functionalized.py_impl(DispatchKey.CompositeExplicitAutograd)
def auto_functionalized_dense(
    op: torch._ops.OpOverload, mutated_args_names: List[str], kwargs: Dict[str, Any]
) -> Tuple[Any, Tuple[Tensor, ...]]:
    new_kwargs = dict(**kwargs)
    result = []
    for name in mutated_args_names:
        new_kwargs[name] = (
            clone_preserve_strides(kwargs[name]) if kwargs[name] is not None else None
        )
        result.append(new_kwargs[name])
    out = op(**new_kwargs)
    return out, tuple(result)


@auto_functionalized.py_impl(FakeTensorMode)
def auto_functionalized_fake(
    mode,
    op: torch._ops.OpOverload,
    mutated_args_names: List[str],
    kwargs: Dict[str, Any],
) -> Tuple[Any, Tuple[Tensor, ...]]:
    with mode:
        result = auto_functionalized_dense(op, mutated_args_names, kwargs)
        return result


@auto_functionalized.py_impl(ProxyTorchDispatchMode)
def auto_functionalized_proxy(
    mode,
    op: torch._ops.OpOverload,
    mutated_args_names: List[str],
    kwargs: Dict[str, Any],
) -> Tuple[Any, Tuple[Tensor, ...]]:
    if not mode.enable_tracing:
        return auto_functionalized(op, mutated_args_names, kwargs)

    with disable_proxy_modes_tracing():
        out = auto_functionalized(op, mutated_args_names, kwargs)

    proxy_kwargs = pytree.tree_map(mode.tracer.unwrap_proxy, kwargs)
    out_proxy = mode.tracer.create_proxy(
        "call_function",
        auto_functionalized,
        (op, mutated_args_names, proxy_kwargs),
        {},
    )
    result = track_tensor_tree(out, out_proxy, constant=None, tracer=mode.tracer)
    return result


auto_functionalized.fallthrough(DispatchKey.AutogradCPU)
auto_functionalized.fallthrough(DispatchKey.AutogradCUDA)


def do_auto_functionalize(
    op: torch._ops.OpOverload, args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> Any:
    """Functionalizes a call to op(*args, **kwargs) by emitting a call to
    `outs = auto_functionalized(op, mutated_args_names, normalized_kwargs)`
    and replacing the mutated (args, kwargs) with the corresponding outputs.

    The normalized_kwargs are just the (args, kwargs), but all in kwarg form.
    This makes handling easier for the auto_functionalized HOP.
    """
    from torch._subclasses.functional_tensor import PythonFunctionalizeAPI

    ctx = PythonFunctionalizeAPI()

    # List of the name of args that get mutated (according to the schema)
    mutable_args_names = []
    # All of the (args, kwargs), but all as kwargs. The names for the
    # args come from the schema. This makes it easier for us to work with them.
    normalized_kwargs = {}
    schema = op._schema
    for idx, arg in enumerate(schema.arguments):
        if arg.alias_info is not None and arg.alias_info.is_write:
            mutable_args_names.append(arg.name)
        # NB: torch_dispatch kwargs are the args defined as kwarg-only in the schema
        if arg.name in kwargs:
            normalized_kwargs[arg.name] = kwargs[arg.name]
        else:
            normalized_kwargs[arg.name] = args[idx]

    unwrapped_kwargs = ctx.unwrap_tensors(normalized_kwargs)  # type: ignore[arg-type]
    with ctx.redispatch_to_next():
        unwrapped_actual_out, unwrapped_outs = auto_functionalized(
            op, mutable_args_names, unwrapped_kwargs  # type: ignore[arg-type]
        )
    assert len(unwrapped_outs) == len(mutable_args_names)
    for name, unwrapped_out in zip(mutable_args_names, unwrapped_outs):
        # Can be None if input was `Tensor(a!)?`
        if unwrapped_out is None:
            continue
        assert isinstance(unwrapped_out, torch.Tensor)
        orig_arg = normalized_kwargs[name]
        ctx.replace(orig_arg, unwrapped_out)
        ctx.commit_update(orig_arg)
        ctx.sync(orig_arg)

    return ctx.wrap_tensors(unwrapped_actual_out)
