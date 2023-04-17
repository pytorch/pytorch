import torch
import copy
from .quantizer import (
    OperatorConfig,
    OperatorPatternType,
    QuantizationConfig,
    QuantizationSpec,
    Quantizer,
)
from .qnnpack_quantizer import (
    _TORCH_DTYPE_TO_QDTYPE,
    _is_annotated,
    _get_default_obs_or_fq_ctr,
)
from typing import List, Dict, Optional
from torch.fx import Node
from torch.ao.quantization.observer import (
    HistogramObserver,
    PerChannelMinMaxObserver,
    PlaceholderObserver,
)

__all__ = [
    "X86InductorQuantizer",
    "get_default_x86_inductor_quantization_config",
]

def supported_quantized_operators() -> Dict[str, List[OperatorPatternType]]:
    supported_operators: Dict[str, List[OperatorPatternType]] = {
        "conv2d": [
            [torch.ops.aten.convolution.default],
        ],
    }
    return copy.deepcopy(supported_operators)

def _get_act_obs_or_fq_ctr(quantization_config: Optional[QuantizationConfig]):
    if quantization_config is None:
        return None
    assert quantization_config is not None
    quantization_spec: QuantizationSpec = quantization_config.activation
    qdtype = _TORCH_DTYPE_TO_QDTYPE[quantization_spec.dtype]
    assert quantization_spec.qscheme in [torch.per_tensor_affine]
    if not quantization_spec.is_dynamic:
        return HistogramObserver.with_args(
            dtype=qdtype,
            quant_min=quantization_spec.quant_min,
            quant_max=quantization_spec.quant_max,
            reduce_range=False,
        )
    else:
        # TODO: extend this helper function to support dynamic quantization
        raise Exception("Unsupported quantization_spec for activation: {}".format(quantization_spec))

def _get_weight_obs_or_fq_ctr(quantization_config: Optional[QuantizationConfig]):
    if quantization_config is None:
        return None
    assert quantization_config is not None
    quantization_spec: QuantizationSpec = quantization_config.weight
    qdtype = _TORCH_DTYPE_TO_QDTYPE[quantization_spec.dtype]
    if quantization_spec.qscheme == torch.per_channel_symmetric:
        return PerChannelMinMaxObserver.with_args(
            qscheme=quantization_spec.qscheme,
            dtype=qdtype,
            quant_min=quantization_spec.quant_min,
            quant_max=quantization_spec.quant_max,
        )
    else:
        raise Exception("Unsupported quantization_spec for weight: {}".format(quantization_spec))

def _get_bias_obs_or_fq_ctr(quantization_config: Optional[QuantizationConfig]):
    if quantization_config is None:
        return None
    assert quantization_config is not None
    quantization_spec: QuantizationSpec = quantization_config.bias
    assert quantization_spec.dtype == torch.float, "Only float dtype for bias is supported for bias right now"
    return PlaceholderObserver.with_args(dtype=quantization_spec.dtype)

def get_default_x86_inductor_quantization_config():
    # Copy from x86 default qconfig from torch/ao/quantization/qconfig.py
    act_quantization_spec = QuantizationSpec(
        dtype=torch.uint8,
        quant_min=0,
        quant_max=255,  # reduce_range=False
        qscheme=torch.per_tensor_affine,
        is_dynamic=False,
    )
    weight_quantization_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        qscheme=torch.per_channel_symmetric,
        ch_axis=0,  # 0 corresponding to weight shape = (oc, ic, kh, kw) of conv
        is_dynamic=False,
    )
    bias_quantization_spec = QuantizationSpec(dtype=torch.float)
    quantization_config = QuantizationConfig(
        act_quantization_spec, weight_quantization_spec, bias_quantization_spec
    )
    return quantization_config

def get_supported_x86_inductor_config_and_operators() -> List[OperatorConfig]:
    supported_config_and_operators: List[OperatorConfig] = []
    for quantization_config in [get_default_x86_inductor_quantization_config(), ]:
        ops = supported_quantized_operators()
        for op_string, pattern_list in ops.items():
            supported_config_and_operators.append(
                OperatorConfig(quantization_config, pattern_list)
            )
    return copy.deepcopy(supported_config_and_operators)

def get_supported_config_and_operators() -> List[OperatorConfig]:
    return get_supported_x86_inductor_config_and_operators()

class X86InductorQuantizer(Quantizer):
    supported_config_and_operators = get_supported_config_and_operators()

    def __init__(self):
        super().__init__()
        self.global_config: Optional[QuantizationConfig] = None
        self.operator_type_config: Dict[str, Optional[QuantizationConfig]] = {}

    @classmethod
    def get_supported_operator_for_quantization_config(
        cls, quantization_config: Optional[QuantizationConfig]
    ) -> List[OperatorPatternType]:
        if quantization_config is None:
            all_ops = []
            for _, ops in cls.supported_config_and_operators:
                all_ops.extend(ops)
            return all_ops

        for config, ops in cls.supported_config_and_operators:
            if config == quantization_config:
                return ops
        return []

    def set_global(self, quantization_config: Optional[QuantizationConfig]):
        self.global_config = quantization_config
        return self

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """ just handling global spec for now
        """
        global_config = self.global_config
        ops = self.get_supported_operator_for_quantization_config(global_config)
        # annotate the nodes from last to first since the matching is in the reversed order
        # and fusion operator patterns (conv - relu) can get matched before single operator pattern (conv)
        # and we will mark the matched node with "_annoated" so fusion operator pattern
        # can take precedence over single operator pattern in this way
        for node in reversed(model.graph.nodes):
            self._annotate_conv2d_binary_unary(node, global_config)
            self._annotate_conv2d_binary(node, global_config)
            self._annotate_conv2d_unary(node, global_config)
            self._annotate_conv2d(node, global_config)
        return model

    def _annotate_conv2d_binary_unary(self, node: Node, quantization_config: Optional[QuantizationConfig]) -> None:
        # Conv2d + add + unary op
        supported_unary_node = [torch.ops.aten.relu_.default, torch.ops.aten.relu.default]
        if node.op != "call_function" or node.target not in supported_unary_node:
            return
        unary_node = node
        assert isinstance(unary_node, Node)
        binary_node = unary_node.args[0]
        assert isinstance(binary_node, Node)
        supported_binary_node = [torch.ops.aten.add_.Tensor, torch.ops.aten.add.Tensor]
        if binary_node.op != "call_function" or binary_node.target not in supported_binary_node:
            return
        conv_node_idx = None
        extra_input_node_idx = None
        if (binary_node.args[0].op == "call_function") and (
            binary_node.args[0].target == torch.ops.aten.convolution.default
        ):
            conv_node_idx = 0
            extra_input_node_idx = 1
        elif (binary_node.args[1].op == "call_function") and (
            binary_node.args[1].target == torch.ops.aten.convolution.default
        ):
            conv_node_idx = 1
            extra_input_node_idx = 0
        if (conv_node_idx is None) or (extra_input_node_idx is None):
            return

        conv_node = binary_node.args[conv_node_idx]
        assert isinstance(conv_node, Node)
        if conv_node.op != "call_function" or conv_node.target != torch.ops.aten.convolution.default:
            # No conv node found to be fused with add
            return
        if _is_annotated([unary_node, binary_node, conv_node]):
            return

        conv_node.meta["target_dtype_info"] = {
            "input_act_obs_or_fq_ctr": _get_act_obs_or_fq_ctr(quantization_config),
            "weight_obs_or_fq_ctr": _get_weight_obs_or_fq_ctr(quantization_config),
            "bias_obs_or_fq_ctr": _get_bias_obs_or_fq_ctr(quantization_config),
            "output_act_obs_or_fq_ctr": _get_default_obs_or_fq_ctr(),
            # TODO: validation of weight_index must be set if weight_obs_or_fq_ctr is set
            "weight_index": 1,
            # TODO: validation of bias_index must be set if bias_obs_or_fq_ctr is set
            "bias_index": 2,
            "_annotated": True,
        }
        # TODO(Leslie) Need to insert observer for the extra input node
        # Maybe use "args_act_index"
        binary_node.meta["target_dtype_info"] = {
            "input_act_obs_or_fq_ctr": _get_act_obs_or_fq_ctr(quantization_config),
            "output_act_obs_or_fq_ctr": _get_default_obs_or_fq_ctr(),
            "_annotated": True,
            "args_act_index": [extra_input_node_idx],
        }
        unary_node.meta["target_dtype_info"] = {
            "input_act_obs_or_fq_ctr": _get_default_obs_or_fq_ctr(),
            "output_act_obs_or_fq_ctr": _get_act_obs_or_fq_ctr(quantization_config),
            "_annotated": True,
        }

    def _annotate_conv2d_binary(self, node: Node, quantization_config: Optional[QuantizationConfig]) -> None:
        # Conv2d + add
        supported_binary_node = [torch.ops.aten.add_.Tensor, torch.ops.aten.add.Tensor]
        if node.op != "call_function" or node.target not in supported_binary_node:
            return
        binary_node = node
        assert isinstance(binary_node, Node)

        conv_node_idx = None
        extra_input_node_idx = None
        if (binary_node.args[0].op == "call_function") and (
            binary_node.args[0].target == torch.ops.aten.convolution.default
        ):
            conv_node_idx = 0
            extra_input_node_idx = 1
        elif (binary_node.args[1].op == "call_function") and (
            binary_node.args[1].target == torch.ops.aten.convolution.default
        ):
            conv_node_idx = 1
            extra_input_node_idx = 0
        if (conv_node_idx is None) or (extra_input_node_idx is None):
            return

        conv_node = binary_node.args[conv_node_idx]
        assert isinstance(conv_node, Node)
        if conv_node.op != "call_function" or conv_node.target != torch.ops.aten.convolution.default:
            # No conv node found to be fused with add
            return
        if _is_annotated([binary_node, conv_node]):
            return

        conv_node.meta["target_dtype_info"] = {
            "input_act_obs_or_fq_ctr": _get_act_obs_or_fq_ctr(quantization_config),
            "weight_obs_or_fq_ctr": _get_weight_obs_or_fq_ctr(quantization_config),
            "bias_obs_or_fq_ctr": _get_bias_obs_or_fq_ctr(quantization_config),
            "output_act_obs_or_fq_ctr": _get_default_obs_or_fq_ctr(),
            # TODO: validation of weight_index must be set if weight_obs_or_fq_ctr is set
            "weight_index": 1,
            # TODO: validation of bias_index must be set if bias_obs_or_fq_ctr is set
            "bias_index": 2,
            "_annotated": True,
        }
        # TODO(Leslie) Need to insert observer for the extra input node
        # Maybe use "args_act_index"
        binary_node.meta["target_dtype_info"] = {
            "input_act_obs_or_fq_ctr": _get_act_obs_or_fq_ctr(quantization_config),
            "output_act_obs_or_fq_ctr": _get_act_obs_or_fq_ctr(quantization_config),
            "_annotated": True,
            "args_act_index": [extra_input_node_idx],
        }

    def _annotate_conv2d_unary(self, node: Node, quantization_config: Optional[QuantizationConfig]) -> None:
        supported_unary_node = [torch.ops.aten.relu_.default, torch.ops.aten.relu.default]
        if node.op != "call_function" or node.target not in supported_unary_node:
            return
        unary_node = node
        conv_node = unary_node.args[0]
        assert isinstance(conv_node, Node)
        if conv_node.op != "call_function" or conv_node.target != torch.ops.aten.convolution.default:
            return
        if _is_annotated([unary_node, conv_node]):
            return

        conv_node.meta["target_dtype_info"] = {
            "input_act_obs_or_fq_ctr": _get_act_obs_or_fq_ctr(quantization_config),
            "weight_obs_or_fq_ctr": _get_weight_obs_or_fq_ctr(quantization_config),
            "bias_obs_or_fq_ctr": _get_bias_obs_or_fq_ctr(quantization_config),
            "output_act_obs_or_fq_ctr": _get_default_obs_or_fq_ctr(),
            # TODO: validation of weight_index must be set if weight_obs_or_fq_ctr is set
            "weight_index": 1,
            # TODO: validation of bias_index must be set if bias_obs_or_fq_ctr is set
            "bias_index": 2,
            "_annotated": True,
        }
        unary_node.meta["target_dtype_info"] = {
            "input_act_obs_or_fq_ctr": _get_default_obs_or_fq_ctr(),
            "output_act_obs_or_fq_ctr": _get_act_obs_or_fq_ctr(quantization_config),
            "_annotated": True,
        }

    def _annotate_conv2d(self, node: Node, quantization_config: Optional[QuantizationConfig]) -> None:
        conv_node = node
        if conv_node.op != "call_function" or conv_node.target != torch.ops.aten.convolution.default:
            return
        # skip annotation if it is already annotated
        if _is_annotated([conv_node]):
            return
        conv_node.meta["target_dtype_info"] = {
            "input_act_obs_or_fq_ctr": _get_act_obs_or_fq_ctr(quantization_config),
            "weight_obs_or_fq_ctr": _get_weight_obs_or_fq_ctr(quantization_config),
            "bias_obs_or_fq_ctr": _get_bias_obs_or_fq_ctr(quantization_config),
            "output_act_obs_or_fq_ctr": _get_act_obs_or_fq_ctr(quantization_config),
            # TODO: validation of weight_index must be set if weight_obs_or_fq_ctr is set
            "weight_index": 1,
            # TODO: validation of bias_index must be set if bias_obs_or_fq_ctr is set
            "bias_index": 2,
            "_annotated": True,
        }

    def validate(self, model: torch.fx.GraphModule) -> None:
        pass

    @classmethod
    def get_supported_operators(cls) -> List[OperatorConfig]:
        return cls.supported_config_and_operators
