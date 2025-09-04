import logging
from typing import Any, Callable

import torch
from functorch.compile import make_boxed_func
from torch import fx

from ..backends.common import aot_autograd
from .registry import CompiledFn, register_backend, register_experimental_backend


log = logging.getLogger(__name__)


@register_experimental_backend
def openxla_eval(
    model: fx.GraphModule, fake_tensor_inputs: list[torch.Tensor]
) -> CompiledFn:
    return xla_backend_helper(model, fake_tensor_inputs, boxed=False)


def openxla_eval_boxed(
    model: fx.GraphModule, fake_tensor_inputs: list[torch.Tensor]
) -> Callable[..., Any]:
    return xla_backend_helper(model, fake_tensor_inputs, boxed=True)


def xla_backend_helper(
    model: fx.GraphModule, fake_tensor_inputs: list[torch.Tensor], boxed: bool = False
) -> Callable[..., Any]:
    try:
        import torch_xla.core.dynamo_bridge as bridge
    except ImportError as e:
        raise ImportError(
            "Please follow the instruction in https://github.com/pytorch/xla#pytorchxla to install torch_xla"
        ) from e

    compiled_graph = None

    def fwd(*args: torch.Tensor) -> Any:
        nonlocal model
        nonlocal compiled_graph
        if compiled_graph is None:
            compiled_graph = bridge.extract_compiled_graph(model, args)
            del model
        return compiled_graph(*args)

    return make_boxed_func(fwd) if boxed else fwd


openxla = aot_autograd(
    fw_compiler=openxla_eval_boxed,
)
register_backend(name="openxla", compiler_fn=openxla)
