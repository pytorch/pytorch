from typing import Any, Dict, List, Optional, Tuple, Union

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

    def __init__(self):
        super().__init__("auto_functionalized")

    def __call__(
        self,
        _mutable_op: torch._ops.OpOverload,
        **kwargs: Dict[str, Any],
    ) -> Tuple[Any, Tuple[Tensor, ...]]:
        assert can_auto_functionalize(_mutable_op)
        assert isinstance(kwargs, dict)
        return super().__call__(_mutable_op, **kwargs)


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
    _mutable_op: torch._ops.OpOverload,
    _only_clone_these_tensors: Optional[Tuple[str, ...]] = None,
    **kwargs: Dict[str, Any],
) -> Tuple[Any, Tuple[Tensor, ...]]:
    new_kwargs = dict(**kwargs)
    result = []

    _mutable_args_names = get_mutable_arg_names(_mutable_op)
    for name in _mutable_args_names:
        if (
            _only_clone_these_tensors is not None
            and name not in _only_clone_these_tensors
        ):
            new_kwargs[name] = kwargs[name]
        else:
            new_kwargs[name] = (
                clone_preserve_strides(kwargs[name])
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
    _mutable_op: torch._ops.OpOverload,
    **kwargs: Dict[str, Any],
) -> Tuple[Any, Tuple[Tensor, ...]]:
    with mode:
        result = auto_functionalized_dense(_mutable_op, **kwargs)
        return result


@auto_functionalized.py_impl(ProxyTorchDispatchMode)
def auto_functionalized_proxy(
    mode,
    _mutable_op: torch._ops.OpOverload,
    **kwargs: Dict[str, Any],
) -> Tuple[Any, Tuple[Tensor, ...]]:
    if not mode.enable_tracing:
        return auto_functionalized(_mutable_op, **kwargs)

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


def get_mutable_arg_names(op: torch._ops.OpOverload) -> List[str]:
    """
    Returns the list of argument names that get mutated according to the
    schema.
    """
    mutable_args_names = [
        arg.name
        for arg in op._schema.arguments
        if arg.alias_info is not None and arg.alias_info.is_write
    ]
    return mutable_args_names


def do_auto_functionalize(
    op: torch._ops.OpOverload, args: Tuple[Any, ...], kwargs: Dict[str, Any]
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
    with ctx.redispatch_to_next():
        unwrapped_outs = auto_functionalized(
            op, **unwrapped_kwargs  # type: ignore[arg-type]
        )

    # List of the name of args that get mutated (according to the schema)
    mutable_args_names = get_mutable_arg_names(op)

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
        assert isinstance(unwrapped_out, torch.Tensor)
        orig_arg = normalized_kwargs[name]
        ctx.replace(orig_arg, unwrapped_out)
        ctx.commit_update(orig_arg)
        ctx.sync(orig_arg)

    return ctx.wrap_tensors(unwrapped_actual_out)  # type: ignore[arg-type]


@auto_functionalized.py_functionalize_impl
def auto_functionalized_func(ctx, _mutable_op, **kwargs):
    unwrapped_kwargs = ctx.unwrap_tensors(kwargs)
    with ctx.redispatch_to_next():
        result = auto_functionalized(_mutable_op, **unwrapped_kwargs)
    return ctx.wrap_tensors(result)
