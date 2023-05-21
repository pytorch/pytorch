import torch
import torch.nn.functional as F
import copy
import functools
import operator
from .quantizer import (
    OperatorConfig,
    OperatorPatternType,
    QuantizationConfig,
    QuantizationSpec,
    Quantizer,
    QuantizationAnnotation,
)
from torch.ao.quantization._pt2e.quantizer.utils import (
    get_act_obs_or_fq_ctr,
    get_bias_obs_or_fq_ctr,
    get_weight_obs_or_fq_ctr,
)
from .qnnpack_quantizer import (
    _is_annotated,
)
from typing import Callable, List, Dict, Optional, Set
from torch.fx import Node

__all__ = [
    "X86InductorQuantizer",
    "get_default_x86_inductor_quantization_config",
]

_QUANT_CONFIG_TO_ANNOTATOR = {}


def register_annotator(quantization_configs: List[QuantizationConfig]):
    def decorator(fn: Callable):
        for quantization_config in quantization_configs:
            if quantization_config in _QUANT_CONFIG_TO_ANNOTATOR:
                raise KeyError(
                    f"Annotator for quantization config {quantization_config} is already registered"
                )
            _QUANT_CONFIG_TO_ANNOTATOR[quantization_config] = functools.partial(
                fn, config=quantization_config
            )

    return decorator


def supported_quantized_operators() -> Dict[str, List[OperatorPatternType]]:
    supported_operators: Dict[str, List[OperatorPatternType]] = {
        "conv2d": [
            [torch.nn.Conv2d],
            [F.conv2d],
            # Conv ReLU
            [torch.nn.Conv2d, torch.nn.ReLU],
            [torch.nn.Conv2d, F.relu],
            [F.conv2d, torch.nn.ReLU],
            [F.conv2d, F.relu],
            # Conv Add
            [torch.nn.Conv2d, torch.add],
            [torch.nn.Conv2d, operator.add],
            [F.conv2d, torch.add],
            [F.conv2d, operator.add],
            # Conv Add ReLU
            [torch.nn.Conv2d, torch.add, torch.nn.ReLU],
            [torch.nn.Conv2d, torch.add, F.relu],
            [torch.nn.Conv2d, operator.add, torch.nn.ReLU],
            [torch.nn.Conv2d, operator.add, F.relu],
            [F.conv2d, torch.add, torch.nn.ReLU],
            [F.conv2d, torch.add, F.relu],
            [F.conv2d, operator.add, torch.nn.ReLU],
            [F.conv2d, operator.add, F.relu],
        ],
    }
    return copy.deepcopy(supported_operators)


def get_supported_x86_inductor_config_and_operators() -> List[OperatorConfig]:
    supported_config_and_operators: List[OperatorConfig] = []
    for quantization_config in [get_default_x86_inductor_quantization_config(), ]:
        ops = supported_quantized_operators()
        for op_string, pattern_list in ops.items():
            supported_config_and_operators.append(
                OperatorConfig(quantization_config, pattern_list)
            )
    return copy.deepcopy(supported_config_and_operators)


@functools.lru_cache
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


def get_supported_config_and_operators() -> List[OperatorConfig]:
    return get_supported_x86_inductor_config_and_operators()


class X86InductorQuantizer(Quantizer):
    supported_config_and_operators = get_supported_config_and_operators()

    def __init__(self):
        super().__init__()
        self.global_config: QuantizationConfig = None  # type: ignore[assignment]
        self.operator_type_config: Dict[str, Optional[QuantizationConfig]] = {}

    @classmethod
    def get_supported_quantization_configs(cls) -> List[QuantizationConfig]:
        op_configs: Set[QuantizationConfig] = set({})
        for spec, _ in cls.supported_config_and_operators:
            op_configs.add(spec)
        return list(op_configs)

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

    def set_global(self, quantization_config: QuantizationConfig):
        self.global_config = quantization_config
        return self

    def set_config_for_operator_type(
        self, operator_type: str, quantization_config: QuantizationConfig
    ):
        self.operator_type_config[operator_type] = quantization_config
        return self

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """ just handling global spec for now
        """
        global_config = self.global_config
        _QUANT_CONFIG_TO_ANNOTATOR[global_config](self, model)

        return model

    @register_annotator(
        [
            get_default_x86_inductor_quantization_config(),
        ]
    )
    def annotate_symmetric_config(
        self, model: torch.fx.GraphModule, config: QuantizationConfig
    ) -> torch.fx.GraphModule:
        # annotate the nodes from last to first since the matching is in the reversed order
        # and fusion operator patterns (conv - relu) can get matched before single operator pattern (conv)
        # and we will mark the matched node with "_annoated" so fusion operator pattern
        # can take precedence over single operator pattern in this way
        for node in reversed(model.graph.nodes):
            # one improvement is to register node annotators for each
            # supported op type.
            self._annotate_conv2d_binary_unary(node, config)
            self._annotate_conv2d_binary(node, config)
            self._annotate_conv2d_unary(node, config)
            self._annotate_conv2d(node, config)
        return model

    def _annotate_conv2d_binary_unary(self, node: Node, quantization_config: QuantizationConfig) -> None:
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
        extra_input_node = binary_node.args[extra_input_node_idx]
        assert isinstance(conv_node, Node)
        if conv_node.op != "call_function" or conv_node.target != torch.ops.aten.convolution.default:
            # No conv node found to be fused with add
            return
        if _is_annotated([unary_node, binary_node, conv_node]):
            return

        input_qspec_map = {}
        input_node = conv_node.args[0]
        assert isinstance(input_node, Node)
        input_qspec_map[input_node] = get_act_obs_or_fq_ctr(quantization_config)

        weight_node = conv_node.args[1]
        assert isinstance(weight_node, Node)
        input_qspec_map[weight_node] = get_weight_obs_or_fq_ctr(quantization_config)

        bias_node = conv_node.args[2]
        if isinstance(bias_node, Node):
            input_qspec_map[bias_node] = get_bias_obs_or_fq_ctr(quantization_config)

        conv_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            _annotated=True
        )
        binary_node_input_qspec_map = {}
        binary_node_input_qspec_map[extra_input_node] = get_act_obs_or_fq_ctr(quantization_config)
        binary_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=binary_node_input_qspec_map,
            _annotated=True
        )
        unary_node.meta["quantization_annotation"] = QuantizationAnnotation(
            output_qspec=get_act_obs_or_fq_ctr(quantization_config),  # type: ignore[arg-type]
            _annotated=True
        )

    def _annotate_conv2d_binary(self, node: Node, quantization_config: QuantizationConfig) -> None:
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
        extra_input_node = binary_node.args[extra_input_node_idx]
        assert isinstance(conv_node, Node)
        if conv_node.op != "call_function" or conv_node.target != torch.ops.aten.convolution.default:
            # No conv node found to be fused with add
            return
        if _is_annotated([binary_node, conv_node]):
            return

        input_qspec_map = {}
        input_node = conv_node.args[0]
        assert isinstance(input_node, Node)
        input_qspec_map[input_node] = get_act_obs_or_fq_ctr(quantization_config)

        weight_node = conv_node.args[1]
        assert isinstance(weight_node, Node)
        input_qspec_map[weight_node] = get_weight_obs_or_fq_ctr(quantization_config)

        bias_node = conv_node.args[2]
        if isinstance(bias_node, Node):
            input_qspec_map[bias_node] = get_bias_obs_or_fq_ctr(quantization_config)

        conv_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            _annotated=True
        )

        binary_node_input_qspec_map = {}
        binary_node_input_qspec_map[extra_input_node] = get_act_obs_or_fq_ctr(quantization_config)
        binary_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=binary_node_input_qspec_map,
            output_qspec=get_act_obs_or_fq_ctr(quantization_config),  # type: ignore[arg-type]
            _annotated=True
        )

    def _annotate_conv2d_unary(self, node: Node, quantization_config: QuantizationConfig) -> None:
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

        input_qspec_map = {}
        input_node = conv_node.args[0]
        assert isinstance(input_node, Node)
        input_qspec_map[input_node] = get_act_obs_or_fq_ctr(quantization_config)

        weight_node = conv_node.args[1]
        assert isinstance(weight_node, Node)
        input_qspec_map[weight_node] = get_weight_obs_or_fq_ctr(quantization_config)

        bias_node = conv_node.args[2]
        if isinstance(bias_node, Node):
            input_qspec_map[bias_node] = get_bias_obs_or_fq_ctr(quantization_config)
        conv_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            _annotated=True
        )
        unary_node.meta["quantization_annotation"] = QuantizationAnnotation(
            output_qspec=get_act_obs_or_fq_ctr(quantization_config),  # type: ignore[arg-type]
            _annotated=True
        )

    def _annotate_conv2d(self, node: Node, quantization_config: QuantizationConfig) -> None:
        conv_node = node
        if conv_node.op != "call_function" or conv_node.target != torch.ops.aten.convolution.default:
            return
        # skip annotation if it is already annotated
        if _is_annotated([conv_node]):
            return
        input_qspec_map = {}
        input_node = conv_node.args[0]
        assert isinstance(input_node, Node)
        input_qspec_map[input_node] = get_act_obs_or_fq_ctr(quantization_config)

        weight_node = conv_node.args[1]
        assert isinstance(weight_node, Node)
        input_qspec_map[weight_node] = get_weight_obs_or_fq_ctr(quantization_config)

        bias_node = conv_node.args[2]
        if isinstance(bias_node, Node):
            input_qspec_map[bias_node] = get_bias_obs_or_fq_ctr(quantization_config)

        conv_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=get_act_obs_or_fq_ctr(quantization_config),
            _annotated=True
        )

    def validate(self, model: torch.fx.GraphModule) -> None:
        pass

    @classmethod
    def get_supported_operators(cls) -> List[OperatorConfig]:
        return cls.supported_config_and_operators
