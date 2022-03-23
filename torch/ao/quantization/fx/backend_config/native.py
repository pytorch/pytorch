import torch
from .observation_type import ObservationType
import torch.nn.qat as nnqat


def _get_default_op_backend_config(op, dtype_config):
    return {
        "pattern": op,
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            dtype_config,
        ],
    }

def get_native_backend_config_dict():
    """ Get backend for PyTorch Native backend_config_dict (fbgemm/qnnpack)
    """
    ############### START dtype configs ##################

    # weighted op int8 config
    # activation: quint8, weight: qint8, bias: float
    weighted_op_int8_dtype_config = {
        # optional, input activation dtype
        "input_dtype": torch.quint8,
        # optional, weight dtype
        "weight_dtype": torch.qint8,
        # optional, bias dtype
        "bias_dtype": torch.float,
        # optional, output activation dtype
        "output_dtype": torch.quint8
    }

    # input_activation: quint8, output_activation: quint8
    default_op_quint8_dtype_config = {
        # optional, input activation dtype
        "input_dtype": torch.quint8,
        # optional, output activation dtype
        "output_dtype": torch.quint8,
    }

    default_op_fp16_dtype_config = {
        # optional, input activation dtype
        "input_dtype": torch.float16,
        # optional, weight dtype
        "weight_dtype": torch.float16,
        # optional, output activation dtype
        "output_dtype": torch.float16,
    }
    ############### END dtype configs ##################

    # operator (module/functional/torch ops) configs
    linear_module_config = {
        # Please see README under this folder for pattern format
        "pattern": torch.nn.Linear,
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_int8_dtype_config,
        ],
        # the root module for the pattern, used to query the reference quantized module
        # e.g. for a (torch.nn.ReLU, torch.nn.Linear) pattern, the root will be torch.nn.Linear
        "root_module": torch.nn.Linear,
        # the corresponding reference quantized module for the root module
        "reference_quantized_module_for_root": torch.nn.quantized._reference.Linear,
        "qat_module": nnqat.Linear,
    }

    default_op_configs = [
        _get_default_op_backend_config(op, default_op_quint8_dtype_config) for op in [
            torch.nn.ConvTranspose1d,
            torch.nn.ConvTranspose2d,
            torch.nn.ELU,
            torch.nn.LeakyReLU,
            torch.nn.Hardswish,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.Dropout,
            torch.nn.functional.elu,
            torch.nn.functional.hardswish,
            torch.nn.functional.instance_norm,
            torch.nn.functional.layer_norm,
            torch.nn.functional.leaky_relu,
            torch.nn.functional.dropout,
        ]]
    default_op_fp16_configs = [
        _get_default_op_backend_config(op, default_op_fp16_dtype_config) for op in [
            torch.nn.LayerNorm,
            torch.nn.SiLU,
            torch.nn.Mish,
            torch.nn.functional.layer_norm,
            torch.nn.functional.silu,
            torch.nn.functional.mish,
            torch.sum,
        ]]

    return {
        # optional
        "name": "native",
        "configs": [
            linear_module_config,
            *default_op_configs,
            *default_op_fp16_configs,
        ],
    }
