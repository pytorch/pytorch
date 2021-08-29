import torch

from .dynamic_tracing.auto_trace import add_auto_observation, add_auto_convert
from .dynamic_tracing.fusion import get_module_fusion_fqns


def prepare(model, example_inputs, inplace=False, allow_list=None,
            observer_non_leaf_module_list=None,
            prepare_custom_config_dict=None):
    r"""A wrapper around `torch.quantization.prepare` which prepares the
    model for quantization using dynamic tracing. Requires `example_inputs` to build
    the graph before calibration or quantization aware training can proceed.

    TODO(future PR): better docblock
    """
    assert example_inputs is not None, 'example_inputs must be specified'

    # automatically fuse modules
    old_class = model.__class__
    model = add_auto_observation(model, example_inputs)
    module_fusion_fqns = get_module_fusion_fqns(model)
    if len(module_fusion_fqns):
        model = torch.quantization.fuse_modules(model, module_fusion_fqns)
    del model._auto_quant_state
    model.__class__ = old_class

    model = torch.quantization.prepare(
        model, inplace, allow_list, observer_non_leaf_module_list,
        prepare_custom_config_dict)
    assert not inplace
    # TODO: disable observers when doing this, to prevent example_inputs
    # from contributing to calibration. Or, insert module observers after
    # this step.
    model = add_auto_observation(model, example_inputs)
    return model


def convert(
        module, mapping=None, inplace=False, remove_qconfig=True,
        convert_custom_config_dict=None):
    r"""A wrapper around `torch.quantization.convert` which converts the model
    to a quantized form using dynamic tracing.

    TODO(future PR): better docblock
    """
    model = torch.quantization.convert(
        module, mapping, inplace, remove_qconfig, convert_custom_config_dict)
    assert not inplace
    model = add_auto_convert(model)
    return model
