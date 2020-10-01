from .quantize import *
from .observer import *
from .qconfig import *
from .fake_quantize import *
from .fuse_modules import fuse_modules
from .stubs import *
from .quant_type import *
from .quantize_jit import *
from .quantize_fx import *
from .quantization_mappings import *
from .fuser_method_mappings import *
from .custom_module_class_mappings import *

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
    'quantize', 'quantize_dynamic', 'quantize_qat',
    'prepare', 'convert', 'prepare_qat',
    # Top level API for graph mode quantization on TorchScript
    'quantize_jit', 'quantize_dynamic_jit',
    # Top level API for graph mode quantization on GraphModule(torch._fx)
    'fuse_fx', 'quantize_fx',  # TODO: add quantize_dynamic_fx
    'prepare_fx', 'prepare_dynamic_fx', 'convert_fx',
    'QuantType',  # quantization type
    # custom module APIs
    'register_static_quant_module_mapping',
    'get_static_quant_module_mappings', 'get_static_quant_module_class',
    'register_dynamic_quant_module_mapping',
    'get_dynamic_quant_module_mappings',
    'register_qat_module_mapping',
    'get_qat_module_mappings',
    'get_qconfig_propagation_list',
    'get_compare_output_module_list',
    'register_quantized_operator_mapping', 'get_quantized_operator',
    'register_fuser_method', 'get_fuser_method',
    'register_observed_custom_module_mapping',
    'get_observed_custom_module_class',
    'register_quantized_custom_mdoule_mapping',
    'get_quantized_custom_module_class',
    'is_custom_module_class',
    'is_observed_custom_module',
    # Sub functions for `prepare` and `swap_module`
    'propagate_qconfig_', 'add_quant_dequant', 'add_observer_', 'swap_module',
    'default_eval_fn', 'get_observer_dict',
    'register_activation_post_process_hook',
    # Observers
    'ObserverBase', 'WeightObserver', 'observer', 'default_observer',
    'default_weight_observer',
    # QConfig
    'QConfig', 'default_qconfig', 'default_dynamic_qconfig', 'float16_dynamic_qconfig',
    # QAT utilities
    'default_qat_qconfig', 'prepare_qat', 'quantize_qat',
    # module transformations
    'fuse_modules',
]
