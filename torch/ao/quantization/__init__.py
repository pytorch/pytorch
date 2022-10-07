# flake8: noqa: F403
# type: ignore[no-redef]

from .fake_quantize import *
from .fuse_modules import fuse_modules
from .fuse_modules import fuse_modules_qat
from .fuser_method_mappings import *
from .observer import *
from .qconfig import *
from .qconfig_mapping import *
from .quant_type import *
from .quantization_mappings import *
from .quantize import *
from .quantize_jit import *
from .stubs import *

def default_eval_fn(model, calib_data):
    r"""
    Default evaluation function takes a torch.utils.data.Dataset or a list of
    input Tensors and run the model on the dataset
    """
    for data, target in calib_data:
        model(data)
