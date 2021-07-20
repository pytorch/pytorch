import torch

from .dynamic_tracing.auto_trace import add_auto_observation, add_auto_convert


def prepare(model, inplace=False, allow_list=None,
            observer_non_leaf_module_list=None,
            prepare_custom_config_dict=None):
    r"""A wrapper around `torch.quantization.prepare` which applies dynamic
    tracing around the prepared model.

    TODO(future PR): better docblock
    """
    model = torch.quantization.prepare(
        model, inplace, allow_list, observer_non_leaf_module_list,
        prepare_custom_config_dict)
    assert not inplace
    model = add_auto_observation(model)
    return model


def convert(
        module, mapping=None, inplace=False, remove_qconfig=True,
        convert_custom_config_dict=None):
    r"""A wrapper around `torch.quantization.convert` which applies dynamic
    tracing around the converted model.

    TODO(future PR): better docblock
    """
    model = torch.quantization.convert(
        module, mapping, inplace, remove_qconfig, convert_custom_config_dict)
    assert not inplace
    model = add_auto_convert(model)
    return model
