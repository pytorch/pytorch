import threading
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import torch


# See [Note: Metadata mutation in proxy tracing] for why sacrificial parameter mutates
# metadata during proxy tracing and we should remove the sacrificial parameter logic.
doc = """
This is used when dynamo traces torch.nn.Parameter, which normally would not trace properly
with AOTAutograd.  We instead create a placeholder torch.nn.Parameter before the graph, which
becomes a graph arg and has no storage backing it.  At the point in the graph where the parameter
actually should be created we mutate this sacrificial placeholder into it.  This allows gradients
to flow into the parameter as if it were an input to the graph (which is the only thing we are
allowed to compute gradients on).
""".strip()


class TracableCreateParameter(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore  # bad-override
    def forward(ctx: Any, tensor: Any, placeholder: Any) -> torch.nn.Parameter:
        assert not tensor.requires_grad
        return placeholder.set_(tensor)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple[None, torch.Tensor]:
        grad = grad_outputs[0]
        return None, grad  # grad flows to placeholder


def tracable_create_parameter(
    tensor: torch.Tensor, placeholder: torch.nn.Parameter
) -> torch.nn.Parameter:
    with torch.set_grad_enabled(placeholder.requires_grad):
        out = TracableCreateParameter.apply(tensor, placeholder)
    return out


def new_parameter_placeholder(
    size: tuple[int, ...], dtype: torch.dtype, device: torch.device, requires_grad: bool
) -> torch.nn.Parameter:
    """Create a placeholder to be passed to the above functions"""
    result = torch.nn.Parameter(
        torch.empty(size, dtype=dtype, device=device), requires_grad=requires_grad
    )
    # TODO(jansel): alloc followed by free is inefficient, need a way to allocate an unbacked tensor.
    # Allocating a zero tensor would causes assert failures in autograd.
    result.untyped_storage().resize_(0)
    return result


_TLS = threading.local()


@contextmanager
def do_not_convert_to_tracable_parameter() -> Generator[bool, None, None]:
    old_flag = getattr(_TLS, "convert_tracable_parameter", True)
    _TLS.convert_tracable_parameter = False
    try:
        yield False
    finally:
        _TLS.convert_tracable_parameter = old_flag


def can_convert_to_tracable_parameter() -> bool:
    return getattr(_TLS, "convert_tracable_parameter", True)
