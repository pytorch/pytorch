from typing import Any, Callable, Dict, List, Tuple, Union

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._ops import HigherOrderOperator
from torch._prims_common import clone_preserve_strides
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)


SIDE_EFFECTFUL_OPS = {
    torch.ops.aten.print.default
}


with_effects = HigherOrderOperator("with_effects")
class WithEffects(HigherOrderOperator):
    """
    with_effects(token, op, args, kwargs) -> (new_token, op_results)
    """

    def __init__(self):
        super().__init__("with_effects")

    def __call__(
        self,
        token,
        op: torch._ops.OpOverload,
        *args: Tuple[Any, ...],
        **kwargs: Dict[str, Any],
    ) -> Tuple[Any, ...]:
        assert has_effects(op, args, kwargs)
        assert isinstance(kwargs, dict)
        return super().__call__(token, op, *args, **kwargs)


with_effects = WithEffects()


def has_effects(op, args, kwargs):
    return get_tokenize_key(op, args, kwargs) is not None


def get_tokenize_key(op, args, kwargs):
    if op in SIDE_EFFECTFUL_OPS:
        return op

    for arg in args:
        if isinstance(arg, torch.ScriptObject):
            return arg

    return None


@with_effects.py_impl(DispatchKey.CompositeExplicitAutograd)
def with_effects_dense(
    token: torch.Tensor,
    op: torch._ops.OpOverload,
    *args: Tuple[Any, ...],
    **kwargs: Dict[str, Any],
) -> Tuple[torch.Tensor, ...]:
    out = op(*args, **kwargs)
    new_token = torch.tensor([])
    if isinstance(out, tuple):
        return (new_token, *out)
    return (new_token, out)


@with_effects.py_impl(FakeTensorMode)
def with_effects_fake(
    mode,
    token: torch.Tensor,
    op: torch._ops.OpOverload,
    *args: Tuple[Any, ...],
    **kwargs: Dict[str, Any],
) -> Tuple[torch.Tensor, ...]:
    with mode:
        result = with_effects_dense(token, op, *args, **kwargs)
        return result


@with_effects.py_impl(ProxyTorchDispatchMode)
def with_effects_proxy(
    mode,
    token: torch.Tensor,
    op: torch._ops.OpOverload,
    *args: Tuple[Any, ...],
    **kwargs: Dict[str, Any],
) -> Tuple[torch.Tensor, ...]:
    if not mode.enable_tracing:
        return with_effects(token, op, *args, **kwargs)

    with disable_proxy_modes_tracing():
        out = with_effects(token, op, *args, **kwargs)

    proxy_token = mode.tracer.unwrap_proxy(token)
    proxy_args = pytree.tree_map(mode.tracer.unwrap_proxy, args)
    proxy_kwargs = pytree.tree_map(mode.tracer.unwrap_proxy, kwargs)

    out_proxy = mode.tracer.create_proxy(
        "call_function",
        with_effects,
        (proxy_token, op, *proxy_args),
        proxy_kwargs,
    )
    result = track_tensor_tree(out, out_proxy, constant=None, tracer=mode.tracer)
    return result


with_effects.fallthrough(DispatchKey.AutogradCPU)
with_effects.fallthrough(DispatchKey.AutogradCUDA)


def handle_effects(
    tokens: Dict[Any, torch.Tensor],
    op: torch._ops.OpOverload,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Tuple[List[torch.Tensor], Any]:
    # Get a token. We can't do `tokens.get(op, torch.tensor([]))` because
    # this will create an empty tensor during proxy mode tracing if the token
    # doesn't exist. But the tokens should always exist during proxy mode tracing.
    key = get_tokenize_key(op, args, kwargs)
    if key not in tokens:
        tokens[key] = torch.tensor([])
    token = tokens[key]

    from torch._subclasses.functional_tensor import PythonFunctionalizeAPI

    ctx = PythonFunctionalizeAPI()

    unwrapped_token = ctx.unwrap_tensors([token])[0]  # type: ignore[arg-type]
    unwrapped_args = ctx.unwrap_tensors(args)  # type: ignore[arg-type]
    unwrapped_kwargs = ctx.unwrap_tensors(kwargs)  # type: ignore[arg-type]
    with ctx.redispatch_to_next():
        (new_token, *unwrapped_outs) = with_effects(
            unwrapped_token, op, *unwrapped_args, **unwrapped_kwargs  # type: ignore[arg-type]
        )

    if len(op._schema.returns) == 0:
        assert unwrapped_outs[0] is None
        unwrapped_outs = None
    elif len(op._schema.returns) == 1:
        assert len(unwrapped_outs) == 1
        unwrapped_outs = unwrapped_outs[0]
    else:
        assert len(unwrapped_outs) == len(op._schema.returns)

    # Add the newly created token into the tokens map for a following call to
    # use this token.
    tokens[key] = new_token

    return ctx.wrap_tensors(unwrapped_outs)  # type: ignore[arg-type]
