import torch
import torch.nn as nn
import torch.nn.intrinsic as nni
import torch.nn.functional as F
import torch.nn.quantized._reference as nnqr
from ._common_operator_config_utils import (
    _get_conv_configs,
    _get_linear_configs,
)
from .backend_config import (
    BackendPatternConfig,
    BackendConfig,
    DTypeConfig,
    ObservationType,
)
from ..fuser_method_mappings import (
    _reverse_sequential_wrapper2,
    _reverse3,
)


# ===================
# |  DTYPE CONFIGS  |
# ===================

onednn_weighted_op_int8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float,
)

onednn_default_dynamic_int8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.float,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float,
    is_dynamic=True,
)

conv_dtype_configs = [onednn_weighted_op_int8_dtype_config]
linear_dtype_configs = [
    onednn_weighted_op_int8_dtype_config,
    onednn_default_dynamic_int8_dtype_config,
]

# ===================
# |  FUSER METHODS  |
# ===================

def _fuse_linear_bn_leaky_relu(is_qat, linear, bn, leaky_relu):
    r"""Given the linear, bn and leaky_relu modules, fuses them and returns the fused module
    Args:
        is_qat: a flag for whether we are using quantization aware training fusion
                or post training quantization fusion
        linear: Module instance of type Linear
        bn: BatchNorm1d instance that needs to be fused with the linear layer
        leaky_relu: LeakyReLU instance that needs to be fused with the linear layer
    Examples::
        >>> m1 = nn.Linear(20, 10)
        >>> b1 = nn.BatchNorm1d(10)
        >>> lr = nn.LeakyReLU(0.01)
        >>> m2 = _fuse_linear_bn_leaky_relu(m1, b1, lr)
    """
    assert(linear.training == bn.training and bn.training == leaky_relu.training),\
        "Linear, BN and LeakyReLU all must be in the same mode (train or eval)."

    if is_qat:
        raise NotImplementedError("Cannot fuse train modules: {}".format((linear, bn, leaky_relu)))
    else:
        map_to_fused_module_eval = {
            nn.Linear: nni.LinearLeakyReLU,
        }
        fused_module = map_to_fused_module_eval.get(type(linear), None)
        if fused_module is not None:
            fused_linear = nn.utils.fusion.fuse_linear_bn_eval(linear, bn)
            fm = fused_module(fused_linear, leaky_relu)
            return fm
        else:
            raise NotImplementedError("Cannot fuse eval modules: {}".format((linear, bn, leaky_relu)))

# ======================
# |  CONFIGS FOR CONV  |
# ======================

conv_configs = _get_conv_configs(conv_dtype_configs)


# ========================
# |  CONFIGS FOR LINEAR  |
# ========================

observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
linear_configs = _get_linear_configs(linear_dtype_configs)

def _add_eltwise_fusion_configs(configs, root_module, root_op, post_module, post_op,
                                dtype_configs, fuser_method, fused_module, observation_type,
                                ref_quant_module):
    # 1 base module + op module fusion config
    configs.append(
        BackendPatternConfig((post_module, root_module))
            .set_dtype_configs(dtype_configs)  # noqa: E131
            .set_fuser_method(fuser_method)
            .set_fused_module(fused_module))
    # base module + functional post op
    configs.append(
        BackendPatternConfig((post_op, root_module))
            .set_dtype_configs(dtype_configs)  # noqa: E131
            .set_fuser_method(fuser_method)
            .set_fused_module(fused_module))

    # 2 fused module configs
    configs.append(
        BackendPatternConfig(fused_module)
            .set_observation_type(observation_type)  # noqa: E131
            .set_dtype_configs(dtype_configs)
            .set_root_module(root_module)
            .set_reference_quantized_module(ref_quant_module))

    # 3 functional base op + post op configs
    configs.append(
        BackendPatternConfig((post_module, root_op))
            .set_observation_type(observation_type)  # noqa: E131
            .set_dtype_configs(dtype_configs))
    configs.append(
        BackendPatternConfig((post_op, root_op))
            .set_observation_type(observation_type)  # noqa: E131
            .set_dtype_configs(dtype_configs))

# Configs for linear + leaky_relu fusion
_add_eltwise_fusion_configs(linear_configs, nn.Linear, F.linear,
                            nn.LeakyReLU, F.leaky_relu, linear_dtype_configs,
                            _reverse_sequential_wrapper2(nni.LinearLeakyReLU),
                            nni.LinearLeakyReLU, observation_type, nnqr.Linear)

# Configs for linear module + batchnorm + leaky_relu
linear_configs.append(
    BackendPatternConfig((nn.LeakyReLU, (nn.BatchNorm1d, nn.Linear)))
        .set_dtype_configs(linear_dtype_configs)  # noqa: E131
        .set_fuser_method(_reverse3(_fuse_linear_bn_leaky_relu))
        .set_fused_module(nni.LinearLeakyReLU))

# Configs for linear + tanh fusion
_add_eltwise_fusion_configs(linear_configs, nn.Linear, F.linear,
                            nn.Tanh, torch.tanh, linear_dtype_configs,
                            _reverse_sequential_wrapper2(nni.LinearTanh),
                            nni.LinearTanh, observation_type, nnqr.Linear)

# =====================
# |  BACKEND CONFIGS  |
# =====================

def get_onednn_backend_config() -> BackendConfig:
    """
    Return the `BackendConfig` for PyTorch's native ONEDNN backend.
    """
    return BackendConfig("onednn") \
        .set_backend_pattern_configs(conv_configs) \
        .set_backend_pattern_configs(linear_configs)

__all__ = [
    "get_onednn_backend_config",
]
