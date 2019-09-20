from __future__ import absolute_import, division, print_function, unicode_literals
from .quantize import *  # noqa: F401
from .quantize_script import *  # noqa: F401
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
    # Top level API for eager mode quantization
    'quantize',
    # Sub functions used by eager mode quantization
    'prepare', 'convert',
    # Sub functions for `prepare` and `swap_module`
    'propagate_qconfig', 'add_quant_dequant', 'add_observer', 'swap_module',
    'default_eval_fn',
    # Top level API for graph mode quantization
    'quantize_script',
    # Sub functions used by graph mode quantization
    'prepare_script', 'convert_script',
    # Observers
    'Observer', 'WeightObserver', 'observer', 'default_observer',
    'default_weight_observer',
    # QConfig
    'QConfig', 'default_qconfig', 'default_dynamic_qconfig',
    # QAT utilities
    'default_qat_qconfig', 'prepare_qat', 'quantize_qat',
    # module transformations
    'fuse_modules',
    # Dynamic quantization utilities
    'quantize_dynamic',
]
