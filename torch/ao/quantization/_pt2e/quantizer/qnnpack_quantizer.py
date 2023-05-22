from __future__ import annotations

import copy
import functools
import operator
from typing import Callable, Dict, List, Optional, Set

import torch
import torch._dynamo as torchdynamo
import torch.nn.functional as F

from torch.ao.quantization._pt2e.quantizer.utils import (
    get_act_obs_or_fq_ctr,
    get_bias_obs_or_fq_ctr,
    get_weight_obs_or_fq_ctr,
)

from torch.ao.quantization.observer import PlaceholderObserver
from torch.fx import Node

from torch.fx.passes.utils.source_matcher_utils import get_source_partitions

from .quantizer import (
    OperatorConfig,
    OperatorPatternType,
    QuantizationConfig,
    QuantizationSpec,
    Quantizer,
    QuantizationAnnotation,
    _annotate_input_qspec_map,
    _annotate_output_qspec,
)

__all__ = [
    "QNNPackQuantizer",
    "get_symmetric_quantization_config",
]

_QUANT_CONFIG_TO_ANNOTATOR = {}


def _mark_nodes_as_annotated(nodes: List[Node]):
    for node in nodes:
        if node is not None:
            if "quantization_annotation" not in node.meta:
                node.meta["quantization_annotation"] = QuantizationAnnotation()
            node.meta["quantization_annotation"]._annotated = True

def _get_dynamo_graph(function: Callable, inputs) -> torch.fx.Graph:
    gm, _ = torchdynamo.export(function, *inputs, aten_graph=True)
    gm.graph.eliminate_dead_code()
    return gm.graph


def _get_linear_patterns(input_size: List[int]):
    in_channels = input_size[-1]
    out_channels = 8  # hard coding but this should not matter
    weight = torch.ones((out_channels, in_channels))
    bias = torch.ones((out_channels,))
    act = torch.ones(input_size)

    def linear_op(act, weight, bias=None):
        return F.linear(act, weight, bias)

    pattern_w_bias = _get_dynamo_graph(linear_op, (act, weight, bias))
    pattern_wo_bias = _get_dynamo_graph(linear_op, (act, weight))
    return [pattern_w_bias, pattern_wo_bias]


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
    }
    return copy.deepcopy(supported_operators)


def get_supported_symmetric_config_and_operators() -> List[OperatorConfig]:
    supported_config_and_operators: List[OperatorConfig] = []
    for quantization_config in [
        get_symmetric_quantization_config(),
        get_symmetric_quantization_config(is_qat=True),
        get_symmetric_quantization_config(is_per_channel=True),
        get_symmetric_quantization_config(is_per_channel=True, is_qat=True),
    ]:
        ops = supported_symmetric_quantized_operators()
        for op_string, pattern_list in ops.items():
            supported_config_and_operators.append(
                OperatorConfig(quantization_config, pattern_list)
            )
    return copy.deepcopy(supported_config_and_operators)


@functools.lru_cache
def get_symmetric_quantization_config(
    is_per_channel: bool = False,
    is_qat: bool = False,
):
    act_quantization_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        qscheme=torch.per_tensor_affine,
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
        ch_axis=0,
        is_dynamic=False,
    )
    bias_quantization_spec = QuantizationSpec(dtype=torch.float)
    quantization_config = QuantizationConfig(
        act_quantization_spec, weight_quantization_spec, bias_quantization_spec, is_qat
    )
    return quantization_config


def get_supported_config_and_operators() -> List[OperatorConfig]:
    return get_supported_symmetric_config_and_operators()


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
            "quantization_annotation" in node.meta
            and node.meta["quantization_annotation"]._annotated
        )
    return annotated


class QNNPackQuantizer(Quantizer):
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
            # note: this assumes each entry in cls.supported_spec_and_operators
            # corresponds to one spec, e.g. we don't have
            # [(spec1, op_list1), (spec1, op_list2), (spec2, op_list3)]
            # where the first and second entry have the same spec but did not
            # merge the op list
            if config == quantization_config:
                return ops
        return []

    def set_global(self, quantization_config: QuantizationConfig) -> QNNPackQuantizer:
        self.global_config = quantization_config
        return self

    def set_config_for_operator_type(
        self, operator_type: str, quantization_config: QuantizationConfig
    ) -> QNNPackQuantizer:
        self.operator_type_config[operator_type] = quantization_config
        return self

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """just handling global spec for now"""
        global_config = self.global_config
        _QUANT_CONFIG_TO_ANNOTATOR[global_config](self, model)

        return model

    @register_annotator(
        [
            get_symmetric_quantization_config(is_per_channel=False, is_qat=False),
            get_symmetric_quantization_config(is_per_channel=False, is_qat=True),
            get_symmetric_quantization_config(is_per_channel=True, is_qat=True),
            get_symmetric_quantization_config(is_per_channel=True, is_qat=False),
        ]
    )
    def annotate_symmetric_config(
        self, model: torch.fx.GraphModule, config: QuantizationConfig
    ) -> torch.fx.GraphModule:
        # annotate the nodes from last to first since the matching is in the reversed order
        # and fusion operator patterns (conv - relu) can get matched before single operator pattern (conv)
        # and we will mark the matched node with "_annoated" so fusion operator pattern
        # can take precedence over single operator pattern in this way
        self._annotate_linear(model, config)
        for node in reversed(model.graph.nodes):
            # one improvement is to register node annotators for each
            # supported op type.
            if config.is_qat:
                self._annotate_conv2d_bn_relu(node, config)
                self._annotate_conv2d_bn(node, config)
            self._annotate_conv2d_relu(node, config)
            self._annotate_conv2d(node, config)
            self._annotate_maxpool2d(node, config)
            self._annotate_add_relu(node, config)
            self._annotate_add(node, config)
            self._annotate_hardtanh(node, config)
            self._annotate_mean(node, config)
            self._annotate_adaptive_avg_pool2d(node, config)
        return model

    def _annotate_conv2d_bn(
        self, node: Node, quantization_config: QuantizationConfig
    ) -> None:
        """
        Match the following pattern:

          ... -> conv -> bn -> getitem[0] -> ...

        Annotate it to get the following pattern after prepare:

               weight -> fq1
                          |
          ...  -> fq0 -> conv -> bn -> getitem[0] -> fq2 -> ...

        Note: This is only used for QAT. In PTQ, batchnorm should already be fused into the conv.
        """
        if (
            node.op != "call_function"
            or node.target != operator.getitem
            or node.args[1] != 0
        ):
            return
        getitem_node = node
        bn_node = getitem_node.args[0]
        assert isinstance(bn_node, Node)
        if (
            bn_node.op != "call_function"
            or bn_node.target != torch.ops.aten._native_batch_norm_legit.default
        ):
            return
        conv_node = bn_node.args[0]
        assert isinstance(conv_node, Node)
        if (
            conv_node.op != "call_function"
            or conv_node.target != torch.ops.aten.convolution.default
        ):
            return
        if _is_annotated([getitem_node, bn_node, conv_node]):
            return

        input_qspec_map = {}
        input_act = conv_node.args[0]
        assert isinstance(input_act, Node)
        input_qspec_map[input_act] = get_act_obs_or_fq_ctr(quantization_config)

        weight = conv_node.args[1]
        assert isinstance(weight, Node)
        input_qspec_map[weight] = get_weight_obs_or_fq_ctr(quantization_config)

        bias = conv_node.args[2]
        if isinstance(bias, Node):
            input_qspec_map[bias] = get_bias_obs_or_fq_ctr(quantization_config)

        conv_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            _annotated=True
        )

        bn_node.meta["quantization_annotation"] = QuantizationAnnotation(
            _annotated=True
        )
        getitem_node.meta["quantization_annotation"] = QuantizationAnnotation(
            output_qspec=get_act_obs_or_fq_ctr(quantization_config),  # type: ignore[arg-type]
            _annotated=True
        )

    def _annotate_conv2d_bn_relu(
        self, node: Node, quantization_config: QuantizationConfig
    ) -> None:
        """
        Match the following pattern:

          ... -> conv -> bn -> getitem[0] -> relu -> ...

        Annotate it to get the following pattern after prepare:

               weight -> fq1
                          |
          ...  -> fq0 -> conv -> bn -> getitem[0] -> relu -> fq2 -> ...

        Note: This is only used for QAT. In PTQ, batchnorm should already be fused into the conv.
        """
        if node.op != "call_function" or node.target not in [
            torch.ops.aten.relu_.default,
            torch.ops.aten.relu.default,
        ]:
            return
        relu_node = node
        getitem_node = relu_node.args[0]
        assert isinstance(getitem_node, Node)
        if (
            getitem_node.op != "call_function"
            or getitem_node.target != operator.getitem
            or getitem_node.args[1] != 0
        ):
            return
        bn_node = getitem_node.args[0]
        assert isinstance(bn_node, Node)
        if (
            bn_node.op != "call_function"
            or bn_node.target != torch.ops.aten._native_batch_norm_legit.default
        ):
            return
        conv_node = bn_node.args[0]
        assert isinstance(conv_node, Node)
        if (
            conv_node.op != "call_function"
            or conv_node.target != torch.ops.aten.convolution.default
        ):
            return
        if _is_annotated([relu_node, getitem_node, bn_node, conv_node]):
            return

        input_qspec_map = {}
        input_act = conv_node.args[0]
        assert isinstance(input_act, Node)
        input_qspec_map[input_act] = get_act_obs_or_fq_ctr(quantization_config)

        weight = conv_node.args[1]
        assert isinstance(weight, Node)
        input_qspec_map[weight] = get_weight_obs_or_fq_ctr(quantization_config)

        bias = conv_node.args[2]
        if isinstance(bias, Node):
            input_qspec_map[bias] = get_bias_obs_or_fq_ctr(quantization_config)

        conv_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            _annotated=True
        )
        bn_node.meta["quantization_annotation"] = QuantizationAnnotation(
            _annotated=True
        )
        getitem_node.meta["quantization_annotation"] = QuantizationAnnotation(
            _annotated=True
        )
        relu_node.meta["quantization_annotation"] = QuantizationAnnotation(
            output_qspec=get_act_obs_or_fq_ctr(quantization_config),  # type: ignore[arg-type]
            _annotated=True
        )

    def _annotate_conv2d_relu(
        self, node: Node, quantization_config: QuantizationConfig
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

        input_qspec_map = {}
        input_act = conv_node.args[0]
        assert isinstance(input_act, Node)
        input_qspec_map[input_act] = get_act_obs_or_fq_ctr(quantization_config)

        weight = conv_node.args[1]
        assert isinstance(weight, Node)
        input_qspec_map[weight] = get_weight_obs_or_fq_ctr(quantization_config)

        bias = conv_node.args[2]
        if isinstance(bias, Node):
            input_qspec_map[bias] = get_bias_obs_or_fq_ctr(quantization_config)

        conv_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            _annotated=True
        )
        relu_node.meta["quantization_annotation"] = QuantizationAnnotation(
            output_qspec=get_act_obs_or_fq_ctr(quantization_config),  # type: ignore[arg-type]
            _annotated=True
        )

    def _annotate_conv2d(
        self, node: Node, quantization_config: QuantizationConfig
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

        input_qspec_map = {}
        input_act = conv_node.args[0]
        assert isinstance(input_act, Node)
        input_qspec_map[input_act] = get_act_obs_or_fq_ctr(quantization_config)

        weight = conv_node.args[1]
        assert isinstance(weight, Node)
        input_qspec_map[weight] = get_weight_obs_or_fq_ctr(quantization_config)

        bias = conv_node.args[2]
        if isinstance(bias, Node):
            input_qspec_map[bias] = get_bias_obs_or_fq_ctr(quantization_config)

        conv_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=get_act_obs_or_fq_ctr(quantization_config),
            _annotated=True
        )

    def _annotate_linear(
        self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig
    ) -> None:
        module_partitions = get_source_partitions(
            gm.graph, [torch.nn.Linear, torch.nn.functional.linear]
        )
        for module_or_fn_type, partitions in module_partitions.items():
            if module_or_fn_type == torch.nn.Linear:
                for p in partitions:
                    act_node = p.input_nodes[0]
                    output_node = p.output_nodes[0]
                    weight_node = None
                    bias_node = None
                    for node in p.params:
                        weight_or_bias = getattr(gm, node.target)  # type: ignore[arg-type]
                        if weight_or_bias.ndim == 2:  # type: ignore[attr-defined]
                            weight_node = node
                        if weight_or_bias.ndim == 1:  # type: ignore[attr-defined]
                            bias_node = node
                    if weight_node is None:
                        raise ValueError("No weight found in Linear pattern")
                    # find use of act node within the matched pattern
                    act_use_node = None
                    for node in p.nodes:
                        if node in act_node.users:  # type: ignore[union-attr]
                            act_use_node = node
                            break
                    if act_use_node is None:
                        raise ValueError(
                            "Could not find an user of act node within matched pattern."
                        )
                    if _is_annotated([act_use_node]) is False:  # type: ignore[list-item]
                        _annotate_input_qspec_map(
                            act_use_node,
                            act_node,
                            get_act_obs_or_fq_ctr(quantization_config),
                        )
                    if bias_node and _is_annotated([bias_node]) is False:
                        _annotate_output_qspec(
                            bias_node, get_bias_obs_or_fq_ctr(quantization_config)
                        )
                    if _is_annotated([weight_node]) is False:  # type: ignore[list-item]
                        _annotate_output_qspec(
                            weight_node, get_weight_obs_or_fq_ctr(quantization_config)
                        )
                    if _is_annotated([output_node]) is False:
                        _annotate_output_qspec(
                            output_node, get_act_obs_or_fq_ctr(quantization_config)
                        )
                    nodes_to_mark_annotated = list(p.nodes)
                    _mark_nodes_as_annotated(nodes_to_mark_annotated)

    # TODO: move to `_pt2e/_propagate_annotation.py` after we have
    # decided on the how we want to use pattern matching for annotation
    def _annotate_maxpool2d(
        self, node: Node, quantization_config: QuantizationConfig
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

        input_act = maxpool_node.args[0]
        assert isinstance(input_act, Node)
        maxpool_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map={
                input_act: get_act_obs_or_fq_ctr(quantization_config)
            },
            _annotated=True,
        )
        getitem_node.meta["quantization_annotation"] = QuantizationAnnotation(
            output_qspec=get_act_obs_or_fq_ctr(quantization_config),
            _input_output_share_observers=True,
            _annotated=True,
        )

    def _annotate_input_out_obs_sharing_op(
        self,
        op: Callable,
        node: Node,
        quantization_config: QuantizationConfig,
    ) -> None:
        io_obs_sharing_node = node
        if (
            io_obs_sharing_node.op != "call_function"
            or io_obs_sharing_node.target != op
        ):
            return
        if _is_annotated([io_obs_sharing_node]):
            return

        input_act = io_obs_sharing_node.args[0]
        assert isinstance(input_act, Node)
        io_obs_sharing_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map={
                input_act: get_act_obs_or_fq_ctr(quantization_config)
            },
            output_qspec=get_act_obs_or_fq_ctr(quantization_config),
            _input_output_share_observers=True,
            _annotated=True,
        )

    def _annotate_hardtanh(
        self, node: Node, quantization_config: QuantizationConfig
    ) -> None:
        self._annotate_input_out_obs_sharing_op(
            torch.ops.aten.hardtanh.default, node, quantization_config
        )

    def _annotate_mean(
        self, node: Node, quantization_config: QuantizationConfig
    ) -> None:
        self._annotate_input_out_obs_sharing_op(
            torch.ops.aten.mean.default, node, quantization_config
        )
        self._annotate_input_out_obs_sharing_op(
            torch.ops.aten.mean.dim, node, quantization_config
        )

    def _annotate_adaptive_avg_pool2d(
        self, node: Node, quantization_config: QuantizationConfig
    ) -> None:
        self._annotate_input_out_obs_sharing_op(
            torch.ops.aten.adaptive_avg_pool2d.default, node, quantization_config
        )

    def _annotate_add_relu(
        self, node: Node, quantization_config: QuantizationConfig
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

        input_qspec_map = {}
        input_act0 = add_node.args[0]
        if isinstance(input_act0, Node):
            input_qspec_map[input_act0] = get_act_obs_or_fq_ctr(quantization_config)

        input_act1 = add_node.args[1]
        if isinstance(input_act1, Node):
            input_qspec_map[input_act1] = get_act_obs_or_fq_ctr(quantization_config)

        add_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            _annotated=True,
        )
        relu_node.meta["quantization_annotation"] = QuantizationAnnotation(
            output_qspec=get_act_obs_or_fq_ctr(quantization_config),
            _annotated=True,
        )

    def _annotate_add(
        self, node: Node, quantization_config: QuantizationConfig
    ) -> None:
        add_node = node
        if add_node.op != "call_function" or add_node.target not in [
            torch.ops.aten.add.Tensor,
            torch.ops.aten.add_.Tensor,
        ]:
            return
        if _is_annotated([add_node]):
            return

        input_qspec_map = {}
        input_act0 = add_node.args[0]
        if isinstance(input_act0, Node):
            input_qspec_map[input_act0] = get_act_obs_or_fq_ctr(quantization_config)

        input_act1 = add_node.args[1]
        if isinstance(input_act1, Node):
            input_qspec_map[input_act1] = get_act_obs_or_fq_ctr(quantization_config)

        add_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=get_act_obs_or_fq_ctr(quantization_config),
            _annotated=True,
        )

    def validate(self, model: torch.fx.GraphModule) -> None:
        pass

    @classmethod
    def get_supported_operators(cls) -> List[OperatorConfig]:
        return cls.supported_config_and_operators
