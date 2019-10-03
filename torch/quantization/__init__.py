from __future__ import absolute_import, division, print_function, unicode_literals
from .quantize import *  # noqa: F401
from .observer import *  # noqa: F401
from .QConfig import *  # noqa: F401
from .fake_quantize import *  # noqa: F401
from .fuse_modules import fuse_modules  # noqa: F401
from .stubs import *  # noqa: F401

def default_eval_fn(model, calib_data):
    r"""
    Default evaluation function takes a torch.utils.data.Dataset or a list of
    input Tensors and run the model on the dataset
    """
    for data, target in calib_data:
        model(data)

_all__ = [
    'QuantWrapper', 'QuantStub', 'DeQuantStub',
    # Top level API for eager mode quantization
    'quantize',
    # Sub functions used by eager mode quantization
    'prepare', 'convert',
    # Sub functions for `prepare` and `swap_module`
    'propagate_qconfig_', 'add_quant_dequant', 'add_observer_', 'swap_module',
    'default_eval_fn', 'get_observer_dict',
    # Observers
    'Observer', 'WeightObserver', 'observer', 'default_observer',
    'default_weight_observer',
    # QConfig
    'QConfig', 'default_qconfig', 'default_dynamic_qconfig', 'float16_dynamic_qconfig',
    # QAT utilities
    'default_qat_qconfig', 'prepare_qat', 'quantize_qat',
    # module transformations
    'fuse_modules',
    # Dynamic quantization utilities
    'quantize_dynamic',
]
