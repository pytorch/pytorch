from .quantize import *  # noqa: F403
# from .observer import *  # noqa: F403
from .qconfig import *  # noqa: F403
from .fake_quantize import *  # noqa: F403
from .fuse_modules import fuse_modules
from .stubs import *  # noqa: F403
from .quant_type import *  # noqa: F403
from .quantize_jit import *  # noqa: F403
# from .quantize_fx import *
from .quantization_mappings import *  # noqa: F403
from .fuser_method_mappings import *  # noqa: F403

import sys
import warnings

from torch.utils._migration_utils import (
    _get_ao_migration_warning_str,
    _AO_MIGRATION_DEPRECATED_NAME_PREFIX,
)

from .observer import _deprecated_names as _observer_deprecated_names
from torch.ao.quantization import observer as __orig_observer_mod
for orig_name in _observer_deprecated_names:
    target_obj_name = f"{_AO_MIGRATION_DEPRECATED_NAME_PREFIX}_{orig_name}"
    target_obj = getattr(__orig_observer_mod, orig_name)
    setattr(sys.modules[__name__], target_obj_name, target_obj)



def default_eval_fn(model, calib_data):
    r"""
    Default evaluation function takes a torch.utils.data.Dataset or a list of
    input Tensors and run the model on the dataset
    """
    for data, target in calib_data:
        model(data)

__all__ = [
    'QuantWrapper', 'QuantStub', 'DeQuantStub',
    # Top level API for eager mode quantization
    'quantize', 'quantize_dynamic', 'quantize_qat',
    'prepare', 'convert', 'prepare_qat',
    # Top level API for graph mode quantization on TorchScript
    'quantize_jit', 'quantize_dynamic_jit', '_prepare_ondevice_dynamic_jit',
    '_convert_ondevice_dynamic_jit', '_quantize_ondevice_dynamic_jit',
    # Top level API for graph mode quantization on GraphModule(torch.fx)
    # 'fuse_fx', 'quantize_fx',  # TODO: add quantize_dynamic_fx
    # 'prepare_fx', 'prepare_dynamic_fx', 'convert_fx',
    'QuantType',  # quantization type
    # custom module APIs
    'get_default_static_quant_module_mappings', 'get_static_quant_module_class',
    'get_default_dynamic_quant_module_mappings',
    'get_default_qat_module_mappings',
    'get_default_qconfig_propagation_list',
    'get_default_compare_output_module_list',
    'get_quantized_operator',
    'get_fuser_method',
    # Sub functions for `prepare` and `swap_module`
    'propagate_qconfig_', 'add_quant_dequant', 'swap_module',
    'default_eval_fn',
    # Observers
    'ObserverBase', 'WeightObserver', 'HistogramObserver',
    'observer', 'default_observer',
    'default_weight_observer', 'default_placeholder_observer',
    'default_per_channel_weight_observer',
    # FakeQuantize (for qat)
    'default_fake_quant', 'default_weight_fake_quant',
    'default_fixed_qparams_range_neg1to1_fake_quant',
    'default_fixed_qparams_range_0to1_fake_quant',
    'default_per_channel_weight_fake_quant',
    'default_histogram_fake_quant',
    # QConfig
    'QConfig', 'default_qconfig', 'default_dynamic_qconfig', 'float16_dynamic_qconfig',
    'float_qparams_weight_only_qconfig',
    # QAT utilities
    'default_qat_qconfig', 'prepare_qat', 'quantize_qat',
    # module transformations
    'fuse_modules',
]

# TODO(this PR): also add from all the other submodules
_deprecated_names = _observer_deprecated_names

def __getattr__(name):
    if name in _deprecated_names:
        warnings.warn(_get_ao_migration_warning_str(__name__, name))
        return globals()[f"{_AO_MIGRATION_DEPRECATED_NAME_PREFIX}_{name}"]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
