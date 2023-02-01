import importlib
import logging

import torch

from torch._dynamo import register_backend
from torch._dynamo.backends.common import dtype_from_inputs, fake_tensor_unsupported

log = logging.getLogger(__name__)


@register_backend
@fake_tensor_unsupported
def ipex(model, inputs, *, dtype=None):
    import intel_extension_for_pytorch as ipex  # type: ignore[import]

    with torch.no_grad():
        model.eval()
        model = ipex.optimize(model, dtype=dtype or dtype_from_inputs(inputs))
        try:
            traced_model = torch.jit.trace(model, inputs).eval()
            traced_model = torch.jit.freeze(traced_model)
            return traced_model
        except Exception:
            log.warning("JIT trace failed during the 'ipex' optimize process.")
            return model


def ipex_fp32(gm: torch.fx.GraphModule, example_inputs):
    return ipex(gm, example_inputs, dtype=torch.float32)


def ipex_bf16(gm: torch.fx.GraphModule, example_inputs):
    return ipex(gm, example_inputs, dtype=torch.bfloat16)


def has_ipex():
    try:
        importlib.import_module("intel_extension_for_pytorch")
        return True
    except ImportError:
        return False
