import importlib
import logging

from torch._dynamo import register_backend

log = logging.getLogger(__name__)


@register_backend
def ipex(model, inputs):
    try:
        import intel_extension_for_pytorch as ipex  # noqa: F401
    except ImportError:
        log.exception(
            "Unable to import Intel Extension for PyTorch (IPEX). "
            "Please install the right version of IPEX that matches the PyTorch version being used. "
            "Refer to https://github.com/intel/intel-extension-for-pytorch for details."
        )
        raise

    return ipex.compile(model, inputs)


def has_ipex():
    try:
        importlib.import_module("intel_extension_for_pytorch")
        return True
    except ImportError:
        return False
