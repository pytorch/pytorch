import importlib
import logging

from torch._dynamo import register_backend

log = logging.getLogger(__name__)


@register_backend
def shark(model, inputs, *, options):
    try:
        from shark.dynamo_backend.utils import SharkBackend
    except ImportError:
        log.exception(
            "Unable to import SHARK - High Performance Machine Learning Distribution"
            "Please install the right version of SHARK that matches the PyTorch version being used. "
            "Refer to https://github.com/nod-ai/SHARK/ for details."
        )
        raise
    return SharkBackend(model, inputs, options)



def has_shark():
    try:
        importlib.import_module("shark")
        return True
    except ImportError:
        return False
