import logging

from ..backends.common import aot_autograd
from ..backends.registry import register_experimental_backend as register_backend

log = logging.getLogger(__name__)


@register_backend
def torchxla_trivial(gm, fake_tensor_inputs):
    return gm


@register_backend
def torchxla_trace_once(model, fake_tensor_inputs):
    import torch_xla.core.dynamo_bridge as bridge  # type: ignore[import]

    compiled_graph = None

    def fwd(*args):
        nonlocal model
        nonlocal compiled_graph
        if compiled_graph is None:
            compiled_graph = bridge.extract_compiled_graph(model, args)
            del model
        return compiled_graph(*args)

    return fwd


aot_torchxla_trivial = aot_autograd(
    fw_compiler=torchxla_trivial,
)
register_backend(name="aot_torchxla_trivial", compiler_fn=aot_torchxla_trivial)

aot_torchxla_trace_once = aot_autograd(
    fw_compiler=torchxla_trace_once,
)
register_backend(name="aot_torchxla_trace_once", compiler_fn=aot_torchxla_trace_once)
