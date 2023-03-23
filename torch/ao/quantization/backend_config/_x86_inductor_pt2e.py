import torch
from torch.ao.quantization.backend_config import (
    BackendConfig,
    DTypeConfig,
    ObservationType,
    BackendPatternConfig,
)
from torch.ao.quantization.utils import MatchAllNode
import itertools

weighted_op_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float,
)

def get_conv_configs():
    conv_configs = []
    observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
    dtype_configs = [weighted_op_quint8_dtype_config]
    conv_configs.append(
        BackendPatternConfig(torch.ops.aten.convolution.default)
        .set_observation_type(observation_type)  # noqa: E131
        .set_dtype_configs(dtype_configs)
        ._set_input_type_to_index({"weight": 1, "bias": 2})
    )
    conv_configs.append(
        BackendPatternConfig((torch.ops.aten.convolution.default, torch.ops.aten.relu.default))
        .set_observation_type(observation_type)  # noqa: E131
        .set_dtype_configs(dtype_configs)
        ._set_input_type_to_index({"weight": 1, "bias": 2})
    )
    # TODO: remove when functionalization is supported in PT2 mode
    conv_configs.append(
        BackendPatternConfig((torch.ops.aten.convolution.default, torch.ops.aten.relu_.default))
        .set_observation_type(observation_type)  # noqa: E131
        .set_dtype_configs(dtype_configs)
        ._set_input_type_to_index({"weight": 1, "bias": 2})
    )

    # Conv add ReLU case
    def _conv_add_relu_root_node_getter_left(pattern):
        relu, add_pattern = pattern
        _, conv, _ = add_pattern
        return conv

    def _conv_add_relu_extra_inputs_getter_left(pattern):
        """ get inputs pattern for extra inputs, inputs for root node
        are assumed to be copied over from root node to the fused node
        """
        relu, add_pattern = pattern
        _, conv, extra_input = add_pattern
        return [extra_input]

    conv_add_relu_optioins = itertools.product(
        [torch.ops.aten.add.Tensor, torch.ops.aten.add_.Tensor],  # add op
        [torch.ops.aten.relu.default, torch.ops.aten.relu_.default],  # relu op
    )

    for add_op, relu_op in conv_add_relu_optioins:
        conv_configs.append(
            BackendPatternConfig()
                ._set_pattern_complex_format((relu_op, (add_op, torch.ops.aten.convolution.default, MatchAllNode)))  # noqa: E131
                .set_observation_type(observation_type)
                .set_dtype_configs(dtype_configs)
                ._set_input_type_to_index({"weight": 1, "bias": 2})
                ._set_root_node_getter(_conv_add_relu_root_node_getter_left)
                ._set_extra_inputs_getter(_conv_add_relu_extra_inputs_getter_left)
        )

    def _conv_add_relu_root_node_getter_right(pattern):
        relu, add_pattern = pattern
        _, _, conv = add_pattern
        return conv

    def _conv_add_relu_extra_inputs_getter_right(pattern):
        """ get inputs pattern for extra inputs, inputs for root node
        are assumed to be copied over from root node to the fused node
        """
        relu, add_pattern = pattern
        _, extra_input, conv = add_pattern
        return [extra_input]

    conv_add_relu_optioins_right = itertools.product(
        [torch.ops.aten.add.Tensor, torch.ops.aten.add_.Tensor],  # add op
        [torch.ops.aten.relu.default, torch.ops.aten.relu_.default],  # relu op
    )

    for add_op, relu_op in conv_add_relu_optioins_right:
        conv_configs.append(
            BackendPatternConfig()
                ._set_pattern_complex_format((relu_op, (add_op, MatchAllNode, torch.ops.aten.convolution.default)))  # noqa: E131
                .set_observation_type(observation_type)
                .set_dtype_configs(dtype_configs)
                ._set_input_type_to_index({"weight": 1, "bias": 2})
                ._set_root_node_getter(_conv_add_relu_root_node_getter_right)
                ._set_extra_inputs_getter(_conv_add_relu_extra_inputs_getter_right)
        )

    # Conv add case
    def _conv_add_root_node_getter_left(pattern):
        _, conv, _ = pattern
        return conv

    def _conv_add_extra_inputs_getter_left(pattern):
        """ get inputs pattern for extra inputs, inputs for root node
        are assumed to be copied over from root node to the fused node
        """
        _, conv, extra_input = pattern
        return [extra_input]

    for add_op in [torch.ops.aten.add.Tensor, torch.ops.aten.add_.Tensor]:
        conv_configs.append(
            BackendPatternConfig()
                ._set_pattern_complex_format((add_op, torch.ops.aten.convolution.default, MatchAllNode))  # noqa: E131
                .set_observation_type(observation_type)
                .set_dtype_configs(dtype_configs)
                ._set_input_type_to_index({"weight": 1, "bias": 2})
                ._set_root_node_getter(_conv_add_root_node_getter_left)
                ._set_extra_inputs_getter(_conv_add_extra_inputs_getter_left)
        )

    def _conv_add_root_node_getter_right(pattern):
        _, _, conv = pattern
        return conv

    def _conv_add_extra_inputs_getter_right(pattern):
        """ get inputs pattern for extra inputs, inputs for root node
        are assumed to be copied over from root node to the fused node
        """
        _, extra_input, conv = pattern
        return [extra_input]

    for add_op in [torch.ops.aten.add.Tensor, torch.ops.aten.add_.Tensor]:
        conv_configs.append(
            BackendPatternConfig()
                ._set_pattern_complex_format((add_op, MatchAllNode, torch.ops.aten.convolution.default))  # noqa: E131
                .set_observation_type(observation_type)
                .set_dtype_configs(dtype_configs)
                ._set_input_type_to_index({"weight": 1, "bias": 2})
                ._set_root_node_getter(_conv_add_root_node_getter_right)
                ._set_extra_inputs_getter(_conv_add_extra_inputs_getter_right)
        )

    return conv_configs

def get_x86_inductor_pt2e_backend_config():
    return (
        BackendConfig("inductor_pytorch_2.0_export")
        .set_backend_pattern_configs(get_conv_configs())
    )
