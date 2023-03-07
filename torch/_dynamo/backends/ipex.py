import importlib
import logging

import torch
from torch._dynamo import register_backend
from .common import fake_tensor_unsupported

log = logging.getLogger(__name__)


@register_backend
@fake_tensor_unsupported
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

    try:
        with torch.no_grad():
            traced_model = torch.jit.trace(model.eval(), inputs)
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
