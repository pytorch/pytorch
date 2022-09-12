# flake8: noqa: F403

from .fake_quantize import *  # noqa: F403
from .fuse_modules import fuse_modules  # noqa: F403
from .fuse_modules import fuse_modules_qat  # noqa: F403
from .fuser_method_mappings import *  # noqa: F403
from .observer import *  # noqa: F403
from .qconfig import *  # noqa: F403
from .qconfig_mapping import *  # noqa: F403
from .quant_type import *  # noqa: F403
from .quantization_mappings import *  # noqa: F403
from .quantize import *  # noqa: F403
from .quantize_jit import *  # noqa: F403
from .stubs import *  # noqa: F403

def default_eval_fn(model, calib_data):
    r"""
    Default evaluation function takes a torch.utils.data.Dataset or a list of
    input Tensors and run the model on the dataset
    """
    for data, target in calib_data:
        model(data)
