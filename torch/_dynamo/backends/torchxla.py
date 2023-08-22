import logging
import warnings

from functorch.compile import make_boxed_func

from ..backends.common import aot_autograd
from ..backends.registry import register_experimental_backend as register_backend

log = logging.getLogger(__name__)


@register_backend
def torchxla_trivial(gm, fake_tensor_inputs):
    return gm


@register_backend
def torchxla_trace_once(model, fake_tensor_inputs):
    warnings.warn(
        "This backend will be deprecated in 2.2, please use `openxla` backend instead",
        FutureWarning,
        stacklevel=2,
    )

    return xla_backend_helper(model, fake_tensor_inputs)


@register_backend
def openxla_eval(model, fake_tensor_inputs):
    return xla_backend_helper(model, fake_tensor_inputs, boxed=False)


def openxla_eval_boxed(model, fake_tensor_inputs):
    return xla_backend_helper(model, fake_tensor_inputs, boxed=True)


def xla_backend_helper(model, fake_tensor_inputs, boxed=False):
    import torch_xla.core.dynamo_bridge as bridge  # type: ignore[import]

    compiled_graph = None

    def fwd(*args):
        nonlocal model
        nonlocal compiled_graph
        if compiled_graph is None:
            compiled_graph = bridge.extract_compiled_graph(model, args)
            del model
        return compiled_graph(*args)

    return make_boxed_func(fwd) if boxed else fwd


aot_torchxla_trivial = aot_autograd(
    fw_compiler=torchxla_trivial,
)
register_backend(name="aot_torchxla_trivial", compiler_fn=aot_torchxla_trivial)

aot_torchxla_trace_once = aot_autograd(
    fw_compiler=torchxla_trace_once,
)
register_backend(name="aot_torchxla_trace_once", compiler_fn=aot_torchxla_trace_once)

openxla = aot_autograd(
    fw_compiler=openxla_eval_boxed,
)
register_backend(name="openxla", compiler_fn=openxla)
