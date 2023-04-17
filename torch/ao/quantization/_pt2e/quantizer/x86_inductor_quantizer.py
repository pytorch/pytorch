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
            self._annotate_conv2d(node, global_config)
        return model

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
