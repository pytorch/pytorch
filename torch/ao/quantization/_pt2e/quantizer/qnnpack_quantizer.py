from __future__ import annotations

import copy
import operator
from dataclasses import asdict
from typing import Callable, Dict, List, Optional, Set

import torch
import torch.nn.functional as F
from torch.ao.quantization.observer import (
    HistogramObserver,
    MinMaxObserver,
    PerChannelMinMaxObserver,
    PlaceholderObserver,
)
from torch.fx import Node

from .quantizer import (
    OperatorConfig,
    OperatorPatternType,
    QuantizationConfig,
    QuantizationSpec,
    Quantizer,
)

__all__ = [
    "QNNPackQuantizer",
    "get_symmetric_quantization_config",
]


def supported_symmetric_quantized_operators() -> Dict[str, List[OperatorPatternType]]:
    supported_operators: Dict[str, List[OperatorPatternType]] = {
        # Both conv and linear should be able to handle relu + hardtanh fusion since
        # those are clamp ops
        "conv2d": [
            [torch.nn.Conv2d, torch.nn.ReLU],
            [torch.nn.Conv2d, F.relu],
            [F.conv2d, torch.nn.ReLU],
            [F.conv2d, F.relu],
        ],
        "linear": [[torch.nn.Linear], [F.linear]],
        "add": [[torch.add]],
        "maxpool2d": [[torch.nn.MaxPool2d], [F.max_pool2d]],
        "hardtanh": [[torch.nn.Hardtanh], [F.hardtanh]],
        "mean": [[torch.mean]],
        "adaptive_avgpool2d": [
            [torch.nn.AdaptiveAvgPool2d],
            [F.adaptive_avg_pool2d],
        ],
        "flatten": [
            [torch.flatten],
        ],
    }
    return copy.deepcopy(supported_operators)


def get_supported_symmetric_config_and_operators() -> List[OperatorConfig]:
    supported_config_and_operators: List[OperatorConfig] = []
    for quantization_config in [
        get_symmetric_quantization_config(),
        get_symmetric_quantization_config(is_per_channel=True),
    ]:
        ops = supported_symmetric_quantized_operators()
        for op_string, pattern_list in ops.items():
            supported_config_and_operators.append(
                OperatorConfig(quantization_config, pattern_list)
            )
    return copy.deepcopy(supported_config_and_operators)


def get_symmetric_quantization_config(is_per_channel=False):
    act_quantization_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        qscheme=torch.per_tensor_symmetric,
        is_dynamic=False,
    )
    qscheme = (
        torch.per_channel_symmetric if is_per_channel else torch.per_tensor_symmetric
    )
    weight_quantization_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-127,
        quant_max=127,
        qscheme=qscheme,
        ch_axis=1,
        is_dynamic=False,
    )
    bias_quantization_spec = QuantizationSpec(dtype=torch.float)
    quantization_config = QuantizationConfig(
        act_quantization_spec, weight_quantization_spec, bias_quantization_spec
    )
    return quantization_config


def get_supported_config_and_operators() -> List[OperatorConfig]:
    return get_supported_symmetric_config_and_operators()


# TODO: add support for torch dtype in quant code base
# this includes observers and prepare/convert code
_TORCH_DTYPE_TO_QDTYPE = {
    torch.int8: torch.qint8,
    torch.uint8: torch.quint8,
    torch.int32: torch.qint32,
    torch.float16: torch.float16,
}


def _get_obs_or_fq_module(
    quantization_spec: QuantizationSpec, extra_kwargs, observer_type
):
    return observer_type.with_args(**asdict(quantization_spec), **extra_kwargs)


def _get_act_obs_or_fq_ctr(quantization_config: Optional[QuantizationConfig]):
    if quantization_config is None:
        return None
    assert quantization_config is not None
    quantization_spec: QuantizationSpec = quantization_config.activation
    qdtype = _TORCH_DTYPE_TO_QDTYPE[quantization_spec.dtype]
    assert quantization_spec.qscheme in [
        torch.per_tensor_affine,
        torch.per_tensor_symmetric,
    ]
    if not quantization_spec.is_dynamic:
        return HistogramObserver.with_args(
            dtype=qdtype,
            quant_min=quantization_spec.quant_min,
            quant_max=quantization_spec.quant_max,
            reduce_range=False,
            eps=2**-12,
        )
    else:
        # TODO: extend this helper function to support dynamic quantization
        raise Exception(
            "Unsupported quantization_spec for activation: {}".format(quantization_spec)
        )


def _get_weight_obs_or_fq_ctr(quantization_config: Optional[QuantizationConfig]):
    if quantization_config is None:
        return None
    assert quantization_config is not None
    quantization_spec: QuantizationSpec = quantization_config.weight
    qdtype = _TORCH_DTYPE_TO_QDTYPE[quantization_spec.dtype]
    if quantization_spec.qscheme == torch.per_tensor_symmetric:
        return MinMaxObserver.with_args(
            qscheme=quantization_spec.qscheme,
            dtype=qdtype,
            quant_min=quantization_spec.quant_min,
            quant_max=quantization_spec.quant_max,
            eps=2**-12,
        )
    elif quantization_spec.qscheme == torch.per_channel_symmetric:
        return PerChannelMinMaxObserver.with_args(
            qscheme=quantization_spec.qscheme,
            dtype=qdtype,
            quant_min=quantization_spec.quant_min,
            quant_max=quantization_spec.quant_max,
            eps=2**-12,
        )
    else:
        raise Exception(
            "Unsupported quantization_spec for weight: {}".format(quantization_spec)
        )


def _get_bias_obs_or_fq_ctr(quantization_config: Optional[QuantizationConfig]):
    if quantization_config is None:
        return None
    assert quantization_config is not None
    quantization_spec: QuantizationSpec = quantization_config.bias
    assert (
        quantization_spec.dtype == torch.float
    ), "Only float dtype for bias is supported for bias right now"
    return PlaceholderObserver.with_args(dtype=quantization_spec.dtype)


def _get_default_obs_or_fq_ctr():
    return PlaceholderObserver.with_args(dtype=torch.float)


def _is_annotated(nodes: List[Node]):
    """
    Given a list of nodes (that represents an operator pattern),
    check if any of the node is annotated, return True if any of the node
    is annotated, otherwise return False
    """
    annotated = False
    for node in nodes:
        annotated = annotated or (
            "target_dtype_info" in node.meta
            and node.meta["target_dtype_info"].get("_annotated", False)
        )
    return annotated


class QNNPackQuantizer(Quantizer):
    supported_config_and_operators = get_supported_config_and_operators()

    def __init__(self):
        super().__init__()
        self.global_config: Optional[QuantizationConfig] = None
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
            # note: this assumes each entry in cls.supported_spec_and_operators
            # corresponds to one spec, e.g. we don't have
            # [(spec1, op_list1), (spec1, op_list2), (spec2, op_list3)]
            # where the first and second entry have the same spec but did not
            # merge the op list
            if config == quantization_config:
                return ops
        return []

    def set_global(
        self, quantization_config: Optional[QuantizationConfig]
    ) -> QNNPackQuantizer:
        self.global_config = quantization_config
        return self

    def set_config_for_operator_type(
        self, operator_type: str, quantization_config: Optional[QuantizationConfig]
    ) -> QNNPackQuantizer:
        self.operator_type_config[operator_type] = quantization_config
        return self

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """just handling global spec for now"""
        global_config = self.global_config
        ops = self.get_supported_operator_for_quantization_config(global_config)
        # annotate the nodes from last to first since the matching is in the reversed order
        # and fusion operator patterns (conv - relu) can get matched before single operator pattern (conv)
        # and we will mark the matched node with "_annoated" so fusion operator pattern
        # can take precedence over single operator pattern in this way
        for node in reversed(model.graph.nodes):
            # one improvement is to register node annotators for each
            # supported op type.
            self._annotate_conv2d_relu(node, global_config)
            self._annotate_conv2d(node, global_config)
            self._annotate_linear(node, global_config)
            self._annotate_maxpool2d(node, global_config)
            self._annotate_add_relu(node, global_config)
            self._annotate_add(node, global_config)

        return model

    def _annotate_conv2d_relu(
        self, node: Node, quantization_config: Optional[QuantizationConfig]
    ) -> None:
        if node.op != "call_function" or node.target not in [
            torch.ops.aten.relu_.default,
            torch.ops.aten.relu.default,
        ]:
            return
        relu_node = node
        conv_node = relu_node.args[0]
        assert isinstance(conv_node, Node)
        if (
            conv_node.op != "call_function"
            or conv_node.target != torch.ops.aten.convolution.default
        ):
            return
        if _is_annotated([relu_node, conv_node]):
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
        relu_node.meta["target_dtype_info"] = {
            "input_act_obs_or_fq_ctr": _get_default_obs_or_fq_ctr(),
            "output_act_obs_or_fq_ctr": _get_act_obs_or_fq_ctr(quantization_config),
            "_annotated": True,
        }

    def _annotate_conv2d(
        self, node: Node, quantization_config: Optional[QuantizationConfig]
    ) -> None:
        conv_node = node
        if (
            conv_node.op != "call_function"
            or conv_node.target != torch.ops.aten.convolution.default
        ):
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

    def _annotate_linear(
        self, node: Node, quantization_config: Optional[QuantizationConfig]
    ) -> None:
        self._annoate_linear3d(node, quantization_config)
        self._annoate_linear2d(node, quantization_config)

    def _annotate_linear3d(
        self, node: Node, quantization_config: Optional[QuantizationConfig]
    ) -> None:
        output_view_node = node
        if (
            output_view_node.op != "call_function" or
            output_view_node.target != torch.ops.aten.view.default
        ):
            return
        addmm_node = output_view_node.args[0]
        assert isinstance(addmm_node, Node)
        if (
            addmm_node.op != "call_function"
            or addmm_node.target != torch.ops.aten.addmm.default
        ):
            return
        input_view_node = addmm_node.args[1]
        assert isinstance(input_view_node, Node)
        if (
            input_view_node.op != "call_function"
            or input_view_node.target != torch.ops.aten.view.default
        ):
            return
        t_node = addmm_node.args[2]
        assert isinstance(t_node, Node)
        if t_node.op != "call_function" or t_node.target != torch.ops.aten.t.default:
            return
        if _is_annotated([output_view_node, addmm_node, input_view_node, t_node]):
            return

        # bias and output act
        addmm_node.meta["target_dtype_info"] = {
            "bias_obs_or_fq_ctr": _get_bias_obs_or_fq_ctr(quantization_config),
            "input_act_obs_or_fq_ctr": _get_default_obs_or_fq_ctr(),
            "output_act_obs_or_fq_ctr": _get_default_obs_or_fq_ctr(),
            "bias_index": 0,
            "_annotated": True,
        }
        # input act
        input_view_node.meta["target_dtype_info"] = {
            "input_act_obs_or_fq_ctr": _get_act_obs_or_fq_ctr(quantization_config),
            "output_act_obs_or_fq_ctr": _get_default_obs_or_fq_ctr(),
            "_annotated": True,
        }
        # weight
        t_node.meta["target_dtype_info"] = {
            "input_act_obs_or_fq_ctr": _get_weight_obs_or_fq_ctr(quantization_config),
            "output_act_obs_or_fq_ctr": _get_default_obs_or_fq_ctr(),
            "_annotated": True,
        }
        # output act
        output_view_node.meta["target_dtype_info"] = {
            "input_act_obs_or_fq_ctr": _get_default_obs_or_fq_ctr(),
            "output_act_obs_or_fq_ctr": _get_act_obs_or_fq_ctr(quantization_config),
            "_annotated": True,
        }

    def _annotate_linear2d(
        self, node: Node, quantization_config: Optional[QuantizationConfig]
    ) -> None:
        addmm_node = node
        if (
            addmm_node.op != "call_function"
            or addmm_node.target != torch.ops.aten.addmm.default
        ):
            return
        t_node = addmm_node.args[2]
        assert isinstance(t_node, Node)
        if t_node.op != "call_function" or t_node.target != torch.ops.aten.t.default:
            return
        if _is_annotated([addmm_node, t_node]):
            return

        # bias, input and output act
        # weight annotated in t_node
        addmm_node.meta["target_dtype_info"] = {
            "bias_obs_or_fq_ctr": _get_bias_obs_or_fq_ctr(quantization_config),
            "input_act_obs_or_fq_ctr": _get_act_obs_or_fq_ctr(quantization_config),
            "weight_obs_or_fq_ctr": _get_default_obs_or_fq_ctr(),
            "output_act_obs_or_fq_ctr": _get_act_obs_or_fq_ctr(quantization_config),
            "bias_index": 0,
            "weight_index": 2,
            "_annotated": True,
        }
        # weight
        t_node.meta["target_dtype_info"] = {
            "input_act_obs_or_fq_ctr": _get_weight_obs_or_fq_ctr(quantization_config),
            "output_act_obs_or_fq_ctr": _get_default_obs_or_fq_ctr(),
            "_annotated": True,
        }

    # TODO: move to `_pt2e/_propagate_annotation.py` after we have
    # decided on the how we want to use pattern matching for annotation
    def _annotate_maxpool2d(
        self, node: Node, quantization_config: Optional[QuantizationConfig]
    ) -> None:
        if (
            node.op != "call_function"
            or node.target != operator.getitem
            or node.args[1] != 0
        ):
            return
        getitem_node = node
        maxpool_node = getitem_node.args[0]
        assert isinstance(maxpool_node, Node)
        if (
            maxpool_node.op != "call_function"
            or maxpool_node.target != torch.ops.aten.max_pool2d_with_indices.default
        ):
            return
        if _is_annotated([getitem_node, maxpool_node]):
            return

        maxpool_node.meta["target_dtype_info"] = {
            "input_act_obs_or_fq_ctr": _get_act_obs_or_fq_ctr(quantization_config),
            "output_act_obs_or_fq_ctr": _get_default_obs_or_fq_ctr(),
            "_annotated": True,
        }
        getitem_node.meta["target_dtype_info"] = {
            "input_act_obs_or_fq_ctr": _get_default_obs_or_fq_ctr(),
            "output_act_obs_or_fq_ctr": _get_act_obs_or_fq_ctr(quantization_config),
            "input_output_share_observers": True,
            "_annotated": True,
        }

    def _annotate_add_relu(
        self, node: Node, quantization_config: Optional[QuantizationConfig]
    ) -> None:
        if node.op != "call_function" or node.target not in [
            torch.ops.aten.relu_.default,
            torch.ops.aten.relu.default,
        ]:
            return
        relu_node = node
        add_node = relu_node.args[0]
        assert isinstance(add_node, Node)
        if add_node.op != "call_function" or add_node.target not in [
            torch.ops.aten.add.Tensor,
            torch.ops.aten.add_.Tensor,
        ]:
            return
        if _is_annotated([relu_node, add_node]):
            return

        add_node.meta["target_dtype_info"] = {
            "input_act_obs_or_fq_ctr": _get_act_obs_or_fq_ctr(quantization_config),
            "output_act_obs_or_fq_ctr": _get_default_obs_or_fq_ctr(),
            "_annotated": True,
        }
        relu_node.meta["target_dtype_info"] = {
            "input_act_obs_or_fq_ctr": _get_default_obs_or_fq_ctr(),
            "output_act_obs_or_fq_ctr": _get_act_obs_or_fq_ctr(quantization_config),
            "_annotated": True,
        }

    def _annotate_add(
        self, node: Node, quantization_config: Optional[QuantizationConfig]
    ) -> None:
        add_node = node
        if add_node.op != "call_function" or add_node.target not in [
            torch.ops.aten.add.Tensor,
            torch.ops.aten.add_.Tensor,
        ]:
            return
        if _is_annotated([add_node]):
            return

        add_node.meta["target_dtype_info"] = {
            "input_act_obs_or_fq_ctr": _get_act_obs_or_fq_ctr(quantization_config),
            "output_act_obs_or_fq_ctr": _get_act_obs_or_fq_ctr(quantization_config),
            "_annotated": True,
        }

    def validate(self, model: torch.fx.GraphModule) -> None:
        pass

    @classmethod
    def get_supported_operators(cls) -> List[OperatorConfig]:
        return cls.supported_config_and_operators
