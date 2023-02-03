import importlib
import logging

import torch

from torch._dynamo import register_backend

log = logging.getLogger(__name__)


@register_backend
def ipex(model, inputs):
    try:
        import intel_extension_for_pytorch  # type: ignore[import]  # noqa: F401
    except ImportError:
        log.exception(
            "Unable to import Intel Extension for PyTorch (IPEX). "
            "Please install the right version of IPEX that matches the PyTorch version being used. "
            "Refer to https://github.com/intel/intel-extension-for-pytorch for details."
        )
        raise

    from torch.utils._mode_utils import no_dispatch

    with no_dispatch():
        static_inputs = []
        for x in inputs:
            if x._has_symbolic_sizes_strides:
                size = [s.node.shape_env.size_hint(s.node.expr) for s in x.size()]
                stride = [s.node.shape_env.size_hint(s.node.expr) for s in x.stride()]
                static_inputs.append(
                    torch.as_strided(
                        torch.zeros(size, dtype=x.dtype, device=x.device), size, stride
                    )
                )
            else:
                static_inputs.append(torch.zeros_like(x))
    try:
        with torch.no_grad():
            traced_model = torch.jit.trace(model.eval(), static_inputs)
            traced_model = torch.jit.freeze(traced_model)
        return traced_model
    except Exception:
        log.warning("JIT trace failed during the 'ipex' optimize process.")
        return model


def has_ipex():
    try:
        importlib.import_module("intel_extension_for_pytorch")
        return True
    except ImportError:
        return False
