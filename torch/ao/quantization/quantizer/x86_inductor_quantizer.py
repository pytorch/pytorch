import copy
import functools
import itertools
import operator
from typing import Any, Dict, List, Optional, Set

import torch
import torch.nn.functional as F
from torch.ao.quantization.observer import (
    HistogramObserver,
    PerChannelMinMaxObserver,
    PlaceholderObserver,
)
from torch.ao.quantization.pt2e.graph_utils import find_sequential_partitions
from torch.ao.quantization.qconfig import _ObserverOrFakeQuantizeConstructor
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (
    _is_annotated,
    get_bias_qspec,
    get_input_act_qspec,
    get_output_act_qspec,
    get_weight_qspec,
    OperatorConfig,
    OperatorPatternType,
    QuantizationConfig,
)
from torch.fx import Node
from torch.fx.passes.utils.source_matcher_utils import (
    get_source_partitions,
    SourcePartition,
)
from .quantizer import QuantizationAnnotation, QuantizationSpec, Quantizer

__all__ = [
    "X86InductorQuantizer",
    "get_default_x86_inductor_quantization_config",
]


def _supported_quantized_operators() -> Dict[str, List[OperatorPatternType]]:
    # TODO: Add more supported operators here.
    supported_operators: Dict[str, List[OperatorPatternType]] = {
        "conv2d": [
            [torch.nn.Conv2d],
            [F.conv2d],
        ],
    }

    # Append Conv Optional(Add) Optioinal(ReLU)
    conv_add_relu_options = itertools.product(
        [torch.nn.Conv2d, F.conv2d],
        [torch.add, operator.add, None],  # add
        [torch.nn.ReLU, F.relu, None],  # relu
    )
    for conv_op, add_op, relu_op in conv_add_relu_options:
        if add_op is None:
            # Append Conv ReLU
            supported_operators["conv2d"].append([conv_op, relu_op])  # type: ignore[list-item]
        elif relu_op is None:
            # Append Conv Add
            supported_operators["conv2d"].append([conv_op, add_op])  # type: ignore[list-item]
        else:
            # Append Conv Add ReLU
            supported_operators["conv2d"].append([conv_op, add_op, relu_op])  # type: ignore[list-item]

    return copy.deepcopy(supported_operators)


def _get_supported_x86_inductor_config_and_operators() -> List[OperatorConfig]:
    supported_config_and_operators: List[OperatorConfig] = []
    for quantization_config in [
        get_default_x86_inductor_quantization_config(),
    ]:
        ops = _supported_quantized_operators()
        for pattern_list in ops.values():
            supported_config_and_operators.append(
                OperatorConfig(quantization_config, pattern_list)
            )
    return copy.deepcopy(supported_config_and_operators)


@functools.lru_cache
def get_default_x86_inductor_quantization_config():
    act_observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor = (
        HistogramObserver
    )

    # Copy from x86 default qconfig from torch/ao/quantization/qconfig.py
    act_quantization_spec = QuantizationSpec(
        dtype=torch.uint8,
        quant_min=0,
        quant_max=255,  # reduce_range=False
        qscheme=torch.per_tensor_affine,
        is_dynamic=False,
        observer_or_fake_quant_ctr=act_observer_or_fake_quant_ctr.with_args(
            eps=2**-12
        ),
    )

    weight_observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor = (
        PerChannelMinMaxObserver
    )
    extra_args: Dict[str, Any] = {"eps": 2**-12}
    weight_quantization_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        qscheme=torch.per_channel_symmetric,
        ch_axis=0,  # 0 corresponding to weight shape = (oc, ic, kh, kw) of conv
        is_dynamic=False,
        observer_or_fake_quant_ctr=weight_observer_or_fake_quant_ctr.with_args(
            **extra_args
        ),
    )
    bias_observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor = (
        PlaceholderObserver
    )
    bias_quantization_spec = QuantizationSpec(
        dtype=torch.float, observer_or_fake_quant_ctr=bias_observer_or_fake_quant_ctr
    )
    quantization_config = QuantizationConfig(
        act_quantization_spec,
        act_quantization_spec,
        weight_quantization_spec,
        bias_quantization_spec,
    )
    return quantization_config


def _get_supported_config_and_operators() -> List[OperatorConfig]:
    return _get_supported_x86_inductor_config_and_operators()


class X86InductorQuantizer(Quantizer):
    supported_config_and_operators = _get_supported_config_and_operators()

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

    def _annotate_conv_node_helper(
        self,
        conv_node: torch.fx.Node,
        annotate_output: bool,
        quantization_config: QuantizationConfig,
    ) -> None:
        """Helper function to annotate the conv node"""
        input_qspec_map = {}
        input_node = conv_node.args[0]
        assert isinstance(input_node, Node)
        input_qspec_map[input_node] = get_input_act_qspec(quantization_config)
        weight_node = conv_node.args[1]
        assert isinstance(weight_node, Node)
        input_qspec_map[weight_node] = get_weight_qspec(quantization_config)
        bias_node = conv_node.args[2]
        if isinstance(bias_node, Node):
            input_qspec_map[bias_node] = get_bias_qspec(quantization_config)
        if annotate_output:
            conv_node.meta["quantization_annotation"] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                # TODO<leslie> Remove the annotate of output when oneDNN qconv support fp32 out.
                output_qspec=get_output_act_qspec(quantization_config),
                _annotated=True,
            )
        else:
            conv_node.meta["quantization_annotation"] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map, _annotated=True
            )

    def _annotate_linear_node_helper(
        self,
        linear_node: torch.fx.Node,
        annotate_output: bool,
        quantization_config: QuantizationConfig,
    ) -> None:
        """Helper function to annotate the linear node"""
        input_qspec_map = {}
        assert linear_node.target in (
            torch.ops.aten.mm.default,
            torch.ops.aten.addmm.default,
        )
        has_bias = linear_node.target is torch.ops.aten.addmm.default
        input_index = 1 if has_bias else 0
        weight_index = input_index + 1

        input_node = linear_node.args[input_index]
        assert isinstance(input_node, Node)
        input_qspec_map[input_node] = get_input_act_qspec(quantization_config)

        t_node = linear_node.args[weight_index]
        assert isinstance(t_node, Node)
        weight_node = t_node.args[0]
        assert isinstance(weight_node, Node)
        quantization_annotation = weight_node.meta.get(
            "quantization_annotation", QuantizationAnnotation()
        )
        quantization_annotation.output_qspec = get_weight_qspec(quantization_config)
        weight_node.meta["quantization_annotation"] = quantization_annotation

        bias_node = linear_node.args[0] if has_bias else None
        if isinstance(bias_node, Node):
            input_qspec_map[bias_node] = get_bias_qspec(quantization_config)

        if annotate_output:
            linear_node.meta["quantization_annotation"] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                # TODO<leslie> Remove the annotate of output
                output_qspec=get_output_act_qspec(quantization_config),
                _annotated=True,
            )
        else:
            linear_node.meta["quantization_annotation"] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map, _annotated=True
            )

    def _get_output_nodes_of_partitions(
        self,
        partition_list: List[SourcePartition],
    ) -> List[torch.fx.Node]:
        """Helper function to get the output node list from partition list"""
        output_node_list = []
        for partition in partition_list:
            if len(partition.output_nodes) > 1:
                raise ValueError("Input partition has more than one output node")
            output_node = partition.output_nodes[0]
            assert isinstance(output_node, Node)
            output_node_list.append(output_node)
        if len(output_node_list) != len(partition_list):
            raise ValueError(
                "length of output_node_list should equal to length of partition_list"
            )
        return output_node_list

    def _get_input_idx_for_binary_node(
        self,
        conv_gemm_node: torch.fx.Node,
        binary_node: torch.fx.Node,
    ):
        """Helper function to check conv_gemm and extra input node index
        for binary node fused with conv_gemm.
        """
        conv_gemm_node_idx = None
        extra_input_node_idx = None
        if (binary_node.args[0].op == "call_function") and (  # type: ignore[union-attr]
            binary_node.args[0] == conv_gemm_node
        ):
            conv_gemm_node_idx = 0
            extra_input_node_idx = 1
        elif (binary_node.args[1].op == "call_function") and (  # type: ignore[union-attr]
            binary_node.args[1] == conv_gemm_node
        ):
            conv_gemm_node_idx = 1
            extra_input_node_idx = 0
        extra_input_node = binary_node.args[extra_input_node_idx]  # type: ignore[index]
        assert isinstance(extra_input_node, Node)
        return conv_gemm_node_idx, extra_input_node_idx

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """just handling global spec for now"""
        model = self._annotate_for_static_quantization_config(model)
        return model

    def _annotate_for_static_quantization_config(
        self, model: torch.fx.GraphModule
    ) -> torch.fx.GraphModule:
        # annotate the nodes from last to first since the matching is in the reversed order
        # and fusion operator patterns (conv - relu) can get matched before single operator pattern (conv)
        # and we will mark the matched node with "_annoated" so fusion operator pattern
        # can take precedence over single operator pattern in this way
        config = self.global_config
        self._annotate_conv2d_binary_unary(model, config)
        self._annotate_conv2d_binary(model, config)
        self._annotate_conv2d_unary(model, config)
        self._annotate_conv2d(model, config)
        self._annotate_linear_unary(model, config)
        self._annotate_linear(model, config)
        return model

    def _annotate_conv2d_binary_unary(
        self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig
    ) -> None:
        # Conv2d + add + unary op
        fused_partitions = find_sequential_partitions(
            gm, [torch.nn.Conv2d, operator.add, torch.nn.ReLU]
        )
        for fused_partition in fused_partitions:
            conv_partition, binary_partition, unary_partition = fused_partition
            conv_node, binary_node, unary_node = self._get_output_nodes_of_partitions(
                [conv_partition, binary_partition, unary_partition]
            )
            conv_node_idx, extra_input_node_idx = self._get_input_idx_for_binary_node(
                conv_node, binary_node
            )
            if (conv_node_idx is None) or (extra_input_node_idx is None):
                continue
            if conv_node != binary_node.args[conv_node_idx]:
                raise ValueError(f"{conv_node} doesn't match input of binary node")
            extra_input_node = binary_node.args[extra_input_node_idx]
            if (
                conv_node.op != "call_function"
                or conv_node.target != torch.ops.aten.convolution.default
            ):
                # No conv node found to be fused with add
                continue
            if _is_annotated([unary_node, binary_node, conv_node]):
                continue
            self._annotate_conv_node_helper(conv_node, False, quantization_config)
            binary_node_input_qspec_map = {}
            binary_node_input_qspec_map[extra_input_node] = get_input_act_qspec(
                quantization_config
            )
            binary_node.meta["quantization_annotation"] = QuantizationAnnotation(
                input_qspec_map=binary_node_input_qspec_map, _annotated=True
            )
            unary_node.meta["quantization_annotation"] = QuantizationAnnotation(
                # TODO<leslie> Remove the annotate of output when oneDNN qconv support fp32 out.
                output_qspec=get_output_act_qspec(quantization_config),  # type: ignore[arg-type]
                _annotated=True,
            )

    def _annotate_conv2d_binary(
        self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig
    ) -> None:
        # Conv2d + add
        fused_partitions = find_sequential_partitions(
            gm, [torch.nn.Conv2d, operator.add]
        )
        for fused_partition in fused_partitions:
            conv_partition, binary_partition = fused_partition
            conv_node, binary_node = self._get_output_nodes_of_partitions(
                [conv_partition, binary_partition]
            )
            conv_node_idx, extra_input_node_idx = self._get_input_idx_for_binary_node(
                conv_node, binary_node
            )
            if (conv_node_idx is None) or (extra_input_node_idx is None):
                continue
            if conv_node != binary_node.args[conv_node_idx]:
                raise ValueError(f"{conv_node} doesn't match input of binary node")
            extra_input_node = binary_node.args[extra_input_node_idx]
            assert isinstance(conv_node, Node)
            if (
                conv_node.op != "call_function"
                or conv_node.target != torch.ops.aten.convolution.default
            ):
                # No conv node found to be fused with add
                continue
            if _is_annotated([binary_node, conv_node]):
                continue
            self._annotate_conv_node_helper(conv_node, False, quantization_config)
            binary_node_input_qspec_map = {}
            binary_node_input_qspec_map[extra_input_node] = get_input_act_qspec(
                quantization_config
            )
            binary_node.meta["quantization_annotation"] = QuantizationAnnotation(
                input_qspec_map=binary_node_input_qspec_map,
                # TODO<leslie> Remove the annotate of output when oneDNN qconv support fp32 out.
                output_qspec=get_output_act_qspec(quantization_config),  # type: ignore[arg-type]
                _annotated=True,
            )

    def _annotate_conv2d_unary(
        self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig
    ) -> None:
        fused_partitions = find_sequential_partitions(
            gm, [torch.nn.Conv2d, torch.nn.ReLU]
        )
        for fused_partition in fused_partitions:
            conv_partition, unary_partition = fused_partition
            conv_node, unary_node = self._get_output_nodes_of_partitions(
                [conv_partition, unary_partition]
            )
            if (
                conv_node.op != "call_function"
                or conv_node.target != torch.ops.aten.convolution.default
            ):
                continue
            if _is_annotated([unary_node, conv_node]):
                continue
            self._annotate_conv_node_helper(conv_node, False, quantization_config)
            unary_node.meta["quantization_annotation"] = QuantizationAnnotation(
                # TODO<leslie> Remove the annotate of output when oneDNN qconv support fp32 out.
                output_qspec=get_output_act_qspec(quantization_config),  # type: ignore[arg-type]
                _annotated=True,
            )

    def _annotate_conv2d(
        self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig
    ) -> None:
        conv_partitions = get_source_partitions(
            gm.graph, [torch.nn.Conv2d, torch.nn.functional.conv2d]
        )
        conv_partitions = list(itertools.chain(*conv_partitions.values()))
        for conv_partition in conv_partitions:
            if len(conv_partition.output_nodes) > 1:
                raise ValueError("conv partition has more than one output node")
            conv_node = conv_partition.output_nodes[0]
            if (
                conv_node.op != "call_function"
                or conv_node.target != torch.ops.aten.convolution.default
            ):
                raise ValueError(f"{conv_node} is not an aten conv2d operator")
            # skip annotation if it is already annotated
            if _is_annotated([conv_node]):
                continue
            self._annotate_conv_node_helper(conv_node, True, quantization_config)

    def _annotate_linear(
        self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig
    ) -> None:
        linear_partitions = get_source_partitions(
            gm.graph, [torch.nn.Linear, torch.nn.functional.linear]
        )
        linear_partitions = list(itertools.chain(*linear_partitions.values()))
        for partition in linear_partitions:
            if len(partition.output_nodes) > 1:
                raise ValueError(
                    "Linear partition cannot have more than one output node"
                )
            linear_node = partition.output_nodes[0]
            if linear_node.op != "call_function" or linear_node.target not in (
                torch.ops.aten.addmm.default,
                torch.ops.aten.mm.default,
            ):
                raise ValueError(f"{linear_node} is not an aten addmm/mm operator")
            # skip annotation if it is already annotated
            if _is_annotated([linear_node]):
                continue
            self._annotate_linear_node_helper(linear_node, True, quantization_config)

    def _annotate_linear_unary(
        self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig
    ) -> None:
        postop_list = [
            torch.nn.ReLU,
            torch.nn.LeakyReLU,
            torch.nn.Tanh,
        ]
        fused_partitions: List[tuple] = []
        for postop in postop_list:
            fused_partitions = fused_partitions + find_sequential_partitions(
                gm, [torch.nn.Linear, postop]
            )
        for fused_partition in fused_partitions:
            linear_partition, unary_partition = fused_partition
            linear_node, unary_node = self._get_output_nodes_of_partitions(
                [linear_partition, unary_partition]
            )
            if linear_node.op != "call_function" or linear_node.target not in (
                torch.ops.aten.addmm.default,
                torch.ops.aten.mm.default,
            ):
                continue
            if _is_annotated([unary_node, linear_node]):
                continue
            self._annotate_linear_node_helper(linear_node, False, quantization_config)
            unary_node.meta["quantization_annotation"] = QuantizationAnnotation(
                # TODO<leslie> Remove the annotate of output when oneDNN qconv support fp32 out.
                output_qspec=get_output_act_qspec(quantization_config),  # type: ignore[arg-type]
                _annotated=True,
            )

    def validate(self, model: torch.fx.GraphModule) -> None:
        pass

    @classmethod
    def get_supported_operators(cls) -> List[OperatorConfig]:
        return cls.supported_config_and_operators
