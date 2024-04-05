from typing import Any, Callable

from .. import _C, autograd


def make_autograd_impl(opdef: Any) -> Callable:
    name: str = f"GeneratedBackwardFor_{opdef._namespace}_{opdef._name}"

    def forward(ctx, *args):
        with _C._AutoDispatchBelowAutograd():
            result = opdef._opoverload(*args)
            if opdef._setup_context_fn:
                opdef._setup_context_fn(ctx, args, result)
            return result

    def backward(ctx, *grads):
        if opdef._backward_fn:
            return opdef._backward_fn(ctx, *grads)
        raise RuntimeError(
            f"Trying to backward through {opdef} but no autograd "
            f"formula was registered. "
            f"Please use register_autograd to add one."
        )

    Generated = type(
        name,
        (autograd.Function,),
        {
            "forward": staticmethod(forward),
            "backward": staticmethod(backward),
        },
    )

    def autograd_impl(*args):
        result = Generated.apply(*args)  # type: ignore[attr-defined]
        return result

    return autograd_impl
