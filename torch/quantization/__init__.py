from __future__ import absolute_import, division, print_function, unicode_literals
from .quantize import *  # noqa: F401
from .observer import *  # noqa: F401
from .QConfig import *  # noqa: F401
from .fake_quantize import *  # noqa: F401
from .fuse_modules import fuse_modules  # noqa: F401

def default_eval_fn(model, calib_data):
    r"""
    Default evaluation function takes a torch.utils.data.Dataset or a list of
    input Tensors and run the model on the dataset
    """
    for data, target in calib_data:
        model(data)

_all__ = [
    'QuantWrapper', 'QuantStub', 'DeQuantStub', 'DEFAULT_MODULE_MAPPING',
    # Top level API for quantizing a float model
    'quantize',
    # Sub functions called by quantize
    'prepare', 'convert',
    # Sub functions for `prepare` and `swap_module`
    'propagate_qconfig', 'add_quant_dequant', 'add_observer', 'swap_module',
    'default_eval_fn',
    # Observers
    'Observer', 'WeightObserver', 'observer', 'default_observer',
    'default_weight_observer',
    # QConfig
    'QConfig', 'default_qconfig',
    # QAT utilities
    'default_qat_qconfig', 'prepare_qat', 'quantize_qat',
    # module transformations
    'fuse_modules',
    # Dynamic quantization utilities
    'quantize_dynamic',
]
