# mypy: allow-untyped-defs
import itertools
import operator
import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Union
from typing_extensions import TypeAlias

import torch
from torch.ao.quantization.pt2e.graph_utils import find_sequential_partitions
from torch.ao.quantization.quantizer.quantizer import (
    QuantizationAnnotation,
    Quantizer,
    SharedQuantizationSpec,
)
from torch.ao.quantization.quantizer.utils import _get_module_name_filter
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (
    get_bias_qspec,
    get_input_act_qspec,
    get_output_act_qspec,
    get_weight_qspec,
    QuantizationConfig,
)
from torch.fx import Node
from torch.fx.passes.utils.source_matcher_utils import (
    get_source_partitions,
    SourcePartition,
)


FilterFn: TypeAlias = Callable[[list[Node]], bool]


__all__ = [
    "OnednnInductorQuantizationAnnotation",
    "QUANT_ANNOTATION_KEY",
    "CurrentQuantizationMode",
    "skip_annotate",
    "create_module_name_filter",
    "create_operator_type_filter",
    "mark_nodes_as_annotated",
    "is_node_annotated",
    "is_any_annotated",
    "is_all_annotated",
    "is_quantized_op_pt2e",
    "annotate_nodes_not_quantize",
    "OnednnInductorQuantizer",
]


@dataclass
class OnednnInductorQuantizationAnnotation(QuantizationAnnotation):
    # _is_output_of_quantized_pattern:
    #  * Node as output node of a fusion pattern.
    #  * The fusion pattern supports int8 data type.
    #  * The fusion pattern has inputs annotated to insert observer.
    #  * The quantization_config is not `None`.
    _is_output_of_quantized_pattern: bool = False


QUANT_ANNOTATION_KEY = "quantization_annotation"


def skip_annotate(nodes: list[Node], filter_fn: Optional[FilterFn] = None) -> bool:
    """Determine whether to skip annotation for a list of nodes."""

    # 1) Skip annotate if any node is already annotated
    if is_any_annotated(nodes):
        return True

    # 2) Proceed annotate if a) a filter function is provided
    # and b) the given nodes list passes the filter function check.
    if filter_fn and filter_fn(nodes):
        return False

    return True


def create_module_name_filter(module_name: str) -> FilterFn:
    """Create a filter function for a given module name.

    The filter function takes a list of nodes (as determined by the annotate function)
    and return True if *all* nodes come from the specified module name, False otherwise.

    For example:
        linear_1: "f32[3, 10]" = torch.ops.aten.linear.default(...) # comes from a module with name `sub.linear1`
        relu: "f32[3, 10]" = torch.ops.aten.relu.default(linear_1); # comes from a module with name `sub.relu1`

    >> module_name_filter = create_module_name_filter_inner("sub")
    >> print(module_name_filter([relu, linear_1]))
    # True  # These two nodes are determined by `_annotate_linear_unary` function and from "sub".
    """

    filter_fn = _get_module_name_filter(module_name)

    def check_all_nodes_from_module(nodes: list[Node]) -> bool:
        all_nodes_from_module_name: bool = all(filter_fn(n) for n in nodes)
        return all_nodes_from_module_name

    return check_all_nodes_from_module


# no change
def create_operator_type_filter(
    operator_type: Callable,
) -> FilterFn:
    """Create a filter function for a given operator type.

    The filter function takes a list of nodes and returns True if it contains
    exactly one node with the specified operator type, False otherwise.

    For example:
        linear_1: "f32[3, 10]" = torch.ops.aten.linear.default(...) # comes from a module with name `sub.linear1`
        relu: "f32[3, 10]" = torch.ops.aten.relu.default(linear_1); # comes from a module with name `sub.relu1`

    >> operator_type_filter = create_operator_type_filter(torch.ops.aten.linear.default)
    >> print(operator_type_filter([relu, linear_1]))
    # True  # These two nodes are determined by `_annotate_linear_unary` function and the second node is `linear`.
    """

    def operator_type_filter(nodes: list[Node]):
        num_nodes_with_operator_type = sum(
            node.target == operator_type for node in nodes
        )
        if num_nodes_with_operator_type > 1:
            raise NotImplementedError(
                f"Several nodes within a single pattern are {operator_type}."
            )
        return num_nodes_with_operator_type == 1

    return operator_type_filter


# no change
def mark_nodes_as_annotated(nodes: list[Node]):
    for node in nodes:
        if node is not None:
            if QUANT_ANNOTATION_KEY not in node.meta:
                node.meta[QUANT_ANNOTATION_KEY] = OnednnInductorQuantizationAnnotation()
            node.meta[QUANT_ANNOTATION_KEY]._annotated = True


# no change
def is_node_annotated(_node):
    """
    return True if the node is annotated, otherwise return False
    """
    return (
        QUANT_ANNOTATION_KEY in _node.meta
        and _node.meta[QUANT_ANNOTATION_KEY]._annotated
    )


# no change
def is_any_annotated(nodes: list[Node]):
    """
    Given a list of nodes (that represents an operator pattern),
    check if any of the node is annotated, return True if any of the node
    is annotated, otherwise return False.
    """
    return any(is_node_annotated(node) for node in nodes)


# no change
def is_all_annotated(nodes: list[Node]):
    """
    Given a list of nodes (that represents an operator pattern),
    return True if all of the node is annotated, otherwise return False.
    """
    return all(is_node_annotated(node) for node in nodes)


def is_quantized_op_pt2e(node: torch.fx.Node):
    """
    Used for pt2e flow to check if the node is a quantized node:
    Case1: the node has been annotated as output node of a fusion pattern.
    Case2: the node has been annotated as single quantized node.
    """
    if not is_any_annotated([node]):
        # The node has not been annotated, directly return False
        return False
    quantization_annotation = node.meta.get(QUANT_ANNOTATION_KEY, None)
    assert isinstance(quantization_annotation, OnednnInductorQuantizationAnnotation)
    return quantization_annotation._is_output_of_quantized_pattern


# no change
def annotate_nodes_not_quantize(nodes: Union[Node, list[Node]]) -> None:
    """Annotate nodes to exclude them from quantization (their `quantization_config` is `None`)."""
    if not isinstance(nodes, list):
        nodes = [nodes]
    for node in nodes:
        node.meta[QUANT_ANNOTATION_KEY] = OnednnInductorQuantizationAnnotation(
            _annotated=True
        )


@dataclass
class CurrentQuantizationMode:
    r"""Configuration defining the current quantization mode for the quantizer.

    All possible current quantization modes are listed below:
    ----------------------------------------------------------------------------------------------------------
                |                                       dynamic_state
     qat_state  |---------------------------------------------------------------------------------------------
                |                           None                              |    True       |  False
    ----------------------------------------------------------------------------------------------------------
        None    | quantizer does not receive a non-None `quantization_config` | \             | \
        False   | quantizer will not do QAT                                   | dynamic       | static
        True    | quantizer will do QAT                                       | QAT + dynamic | QAT + static
    """

    qat_state: Optional[bool]
    dynamic_state: Optional[bool]


class OnednnInductorQuantizer(Quantizer):
    # no change
    def __init__(self) -> None:
        super().__init__()
        self.global_config: Optional[QuantizationConfig] = None
        self.operator_type_qconfig: dict[
            torch._ops.OpOverloadPacket, Optional[QuantizationConfig]
        ] = {}
        self.module_name_qconfig: dict[str, Optional[QuantizationConfig]] = {}

    # no change
    def _get_current_quantization_mode(self) -> CurrentQuantizationMode:
        """Retrieves the current quantization mode based on all configurations."""
        qat_state = None
        dynamic_state = None

        # As we use `_need_skip_config` to skip all invalid configurations,
        # we can safely assume that the all existing non-None configurations
        # have the same quantization mode.
        for qconfig in (
            list(self.module_name_qconfig.values())
            + list(self.operator_type_qconfig.values())
            + [self.global_config]
        ):
            if qconfig is not None:
                # Query the `is_qat` state
                if qat_state is None:
                    qat_state = qconfig.is_qat
                else:
                    assert qat_state == qconfig.is_qat, (
                        f"All non-None quantization configs should have the same `is_qat`,"
                        f"but got {qat_state} and {qconfig.is_qat}."
                    )
                # Query the `is_dynamic` state
                input_activation_spec = qconfig.input_activation
                if input_activation_spec is not None:
                    if dynamic_state is None:
                        dynamic_state = input_activation_spec.is_dynamic
                    else:
                        assert dynamic_state == input_activation_spec.is_dynamic, (
                            f"All non-None `input_activation_spec` should have the same `is_dynamic`,"
                            f"but got {dynamic_state} and {input_activation_spec.is_dynamic}."
                        )
        return CurrentQuantizationMode(qat_state=qat_state, dynamic_state=dynamic_state)

    # no change
    def _need_skip_config(
        self, quantization_config: Optional[QuantizationConfig]
    ) -> bool:
        """Check if the provided quantization config is valid for OnednnInductorQuantizer.

        Mixed static/dynamic configurations or mixed QAT/non-QAT configurations are not supported.
        To avoid such a mix, we compare the incoming configuration with current configuration status.
        Refer the `CurrentQuantizationMode` definition for all possible modes.
        """
        if quantization_config is None:
            return False

        need_skip = False
        current_mode = self._get_current_quantization_mode()
        if (
            current_mode.qat_state is not None
            and current_mode.qat_state != quantization_config.is_qat
        ):
            warnings.warn("Mixed QAT and Non-QAT quantization config is not supported.")
            need_skip = True
        if current_mode.dynamic_state is not None:
            input_activation_spec = quantization_config.input_activation
            if (
                input_activation_spec is not None
                and current_mode.dynamic_state != input_activation_spec.is_dynamic
            ):
                warnings.warn(
                    "Mixed dynamic and static quantization config is not supported."
                )
                need_skip = True
        return need_skip

    # no change
    def set_global(self, quantization_config: QuantizationConfig):
        if self._need_skip_config(quantization_config):
            warnings.warn("Skip the global quantization config.")
            return self
        self.global_config = quantization_config
        return self

    # no change
    def _annotate_conv_node_helper(
        self,
        conv_node: torch.fx.Node,
        annotate_output: bool,
        quantization_config: Optional[QuantizationConfig],
    ) -> None:
        """Helper function to annotate the conv node"""
        if quantization_config is None:
            annotate_nodes_not_quantize(conv_node)
            return
        input_qspec_map = {}
        input_node = conv_node.args[0]
        assert isinstance(input_node, Node)
        input_qspec_map[input_node] = get_input_act_qspec(quantization_config)
        weight_node = conv_node.args[1]
        assert isinstance(weight_node, Node)
        input_qspec_map[weight_node] = get_weight_qspec(quantization_config)
        bias_node = None if len(conv_node.args) == 2 else conv_node.args[2]
        if isinstance(bias_node, Node):
            input_qspec_map[bias_node] = get_bias_qspec(quantization_config)
        if annotate_output:
            conv_node.meta[QUANT_ANNOTATION_KEY] = OnednnInductorQuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                _annotated=True,
                _is_output_of_quantized_pattern=True,
            )
        else:
            conv_node.meta[QUANT_ANNOTATION_KEY] = OnednnInductorQuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                _annotated=True,
            )

    # no change
    def _annotate_linear_node_helper(
        self,
        linear_node: torch.fx.Node,
        annotate_output: bool,
        quantization_config: Optional[QuantizationConfig],
    ) -> None:
        """Helper function to annotate the linear node"""
        if quantization_config is None:
            annotate_nodes_not_quantize(linear_node)
            return
        input_qspec_map = {}
        assert linear_node.target in (torch.ops.aten.linear.default,)
        has_bias = len(linear_node.args) == 3
        input_index = 0
        weight_index = 1
        bias_index = 2

        input_node = linear_node.args[input_index]
        assert isinstance(input_node, Node)
        input_qspec_map[input_node] = get_input_act_qspec(quantization_config)

        weight_node = linear_node.args[weight_index]
        assert isinstance(weight_node, Node)
        input_qspec_map[weight_node] = get_weight_qspec(quantization_config)

        bias_node = linear_node.args[bias_index] if has_bias else None
        if isinstance(bias_node, Node):
            input_qspec_map[bias_node] = get_bias_qspec(quantization_config)

        if annotate_output:
            linear_node.meta[
                QUANT_ANNOTATION_KEY
            ] = OnednnInductorQuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                _annotated=True,
                _is_output_of_quantized_pattern=True,
            )
        else:
            linear_node.meta[
                QUANT_ANNOTATION_KEY
            ] = OnednnInductorQuantizationAnnotation(
                input_qspec_map=input_qspec_map, _annotated=True
            )

    # no change
    def _get_output_nodes_of_partitions(
        self,
        partition_list: list[SourcePartition],
    ) -> list[torch.fx.Node]:
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

    # no change
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

    def _annotate_qat_conv2d_bn_binary(
        self,
        gm: torch.fx.GraphModule,
        quantization_config: Optional[QuantizationConfig],
        filter_fn: Optional[FilterFn] = None,
    ) -> None:
        fused_partitions = find_sequential_partitions(
            gm, [torch.nn.Conv2d, torch.nn.BatchNorm2d, operator.add]
        )
        for fused_partition in fused_partitions:
            conv_partition, bn_partition, binary_partition = fused_partition
            (
                conv_node,
                bn_output_node,
                binary_node,
            ) = self._get_output_nodes_of_partitions(
                [conv_partition, bn_partition, binary_partition]
            )
            if len(bn_output_node.users) != 1:
                # Conv BN pattern should only has 1 user.
                continue
            (
                bn_output_node_idx,
                extra_input_node_idx,
            ) = self._get_input_idx_for_binary_node(bn_output_node, binary_node)
            if (bn_output_node_idx is None) or (extra_input_node_idx is None):
                continue
            if bn_output_node != binary_node.args[bn_output_node_idx]:
                raise ValueError(f"{bn_output_node} doesn't match input of binary node")

            extra_input_node = binary_node.args[extra_input_node_idx]

            if (
                conv_node.op != "call_function"
                or conv_node.target != torch.ops.aten.conv2d.default
            ):
                continue

            if skip_annotate([binary_node, bn_output_node, conv_node], filter_fn):
                continue

            self._annotate_conv_node_helper(conv_node, False, quantization_config)

            if quantization_config is not None:
                binary_node_input_qspec_map = {}
                binary_node_input_qspec_map[extra_input_node] = get_input_act_qspec(
                    quantization_config
                )
                binary_node.meta[
                    QUANT_ANNOTATION_KEY
                ] = OnednnInductorQuantizationAnnotation(
                    input_qspec_map=binary_node_input_qspec_map,
                    # TODO<leslie> Remove the annotate of output in QAT when qat util support pattern matcher.
                    output_qspec=get_output_act_qspec(quantization_config),  # type: ignore[arg-type]
                    _annotated=True,
                    _is_output_of_quantized_pattern=True,
                )
            else:
                annotate_nodes_not_quantize(binary_node)
            nodes_to_mark_annotated = list(conv_partition.nodes)
            nodes_to_mark_annotated.extend(list(bn_partition.nodes))
            nodes_to_mark_annotated.extend(list(binary_partition.nodes))
            mark_nodes_as_annotated(nodes_to_mark_annotated)

    def _annotate_qat_conv2d_bn(
        self,
        gm: torch.fx.GraphModule,
        quantization_config: Optional[QuantizationConfig],
        filter_fn: Optional[FilterFn] = None,
    ) -> None:
        fused_partitions = find_sequential_partitions(
            gm, [torch.nn.Conv2d, torch.nn.BatchNorm2d]
        )
        for fused_partition in fused_partitions:
            conv_partition, bn_partition = fused_partition
            conv_node, bn_output_node = self._get_output_nodes_of_partitions(
                [conv_partition, bn_partition]
            )

            if (
                conv_node.op != "call_function"
                or conv_node.target != torch.ops.aten.conv2d.default
            ):
                continue

            if skip_annotate([bn_output_node, conv_node], filter_fn):
                continue

            self._annotate_conv_node_helper(conv_node, False, quantization_config)
            if quantization_config is not None:
                bn_output_node.meta[
                    QUANT_ANNOTATION_KEY
                ] = OnednnInductorQuantizationAnnotation(
                    # TODO<leslie> Remove the annotate of output in QAT when qat util support pattern matcher.
                    output_qspec=get_output_act_qspec(quantization_config),  # type: ignore[arg-type]
                    _annotated=True,
                    _is_output_of_quantized_pattern=True,
                )
            else:
                annotate_nodes_not_quantize(bn_output_node)
            nodes_to_mark_annotated = list(conv_partition.nodes)
            nodes_to_mark_annotated.extend(list(bn_partition.nodes))
            mark_nodes_as_annotated(nodes_to_mark_annotated)

    def _annotate_matmul(
        self,
        model: torch.fx.GraphModule,
        quantization_config: Optional[QuantizationConfig],
        filter_fn: Optional[FilterFn] = None,
    ):
        for node in model.graph.nodes:
            if node.target != torch.ops.aten.matmul.default:
                continue
            if skip_annotate([node], filter_fn):
                continue

            if quantization_config is None:
                annotate_nodes_not_quantize(node)
                continue

            input_qspec_map = {}
            matmul_node = node
            for input_node in matmul_node.args:
                input_qspec_map[input_node] = get_input_act_qspec(quantization_config)
            matmul_node.meta[
                QUANT_ANNOTATION_KEY
            ] = OnednnInductorQuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                _annotated=True,
                _is_output_of_quantized_pattern=True,
            )

    def _annotate_conv2d_binary(
        self,
        gm: torch.fx.GraphModule,
        quantization_config: Optional[QuantizationConfig],
        filter_fn: Optional[FilterFn] = None,
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
            if len(conv_node.users) != 1:
                # Conv Node should only has 1 user node
                continue
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
                or conv_node.target != torch.ops.aten.conv2d.default
            ):
                # No conv node found to be fused with add
                continue
            if skip_annotate([binary_node, conv_node], filter_fn):
                continue

            if quantization_config is None:
                annotate_nodes_not_quantize([conv_node, binary_node])
                continue

            self._annotate_conv_node_helper(conv_node, False, quantization_config)
            binary_node_input_qspec_map = {}
            binary_node_input_qspec_map[extra_input_node] = get_input_act_qspec(
                quantization_config
            )
            binary_node.meta[
                QUANT_ANNOTATION_KEY
            ] = OnednnInductorQuantizationAnnotation(
                input_qspec_map=binary_node_input_qspec_map,
                _annotated=True,
                _is_output_of_quantized_pattern=True,
            )

    def _annotate_conv2d(
        self,
        gm: torch.fx.GraphModule,
        quantization_config: Optional[QuantizationConfig],
        filter_fn: Optional[FilterFn] = None,
    ) -> None:
        conv_partitions = get_source_partitions(
            gm.graph, [torch.nn.Conv2d, torch.nn.functional.conv2d]
        )
        conv_partitions = list(itertools.chain.from_iterable(conv_partitions.values()))
        for conv_partition in conv_partitions:
            if len(conv_partition.output_nodes) > 1:
                raise ValueError("conv partition has more than one output node")
            conv_node = conv_partition.output_nodes[0]
            if (
                conv_node.op != "call_function"
                or conv_node.target != torch.ops.aten.conv2d.default
            ):
                raise ValueError(f"{conv_node} is not an aten conv2d operator")
            # skip annotation if it is already annotated
            if skip_annotate([conv_node], filter_fn):
                continue
            self._annotate_conv_node_helper(conv_node, True, quantization_config)

    def _annotate_output_share_observer_as_input(
        self, input_node: Node, source_node: Node
    ):
        source_node_quantization_annotation = (
            source_node.meta[QUANT_ANNOTATION_KEY]
            if QUANT_ANNOTATION_KEY in source_node.meta
            else None
        )
        if (
            source_node_quantization_annotation
            and source_node_quantization_annotation._is_output_of_quantized_pattern
        ):
            edge_or_node = (input_node, source_node)
            source_node_quantization_annotation.output_qspec = SharedQuantizationSpec(
                edge_or_node
            )
        return

    def _annotate_linear(
        self,
        gm: torch.fx.GraphModule,
        quantization_config: Optional[QuantizationConfig],
        filter_fn: Optional[FilterFn] = None,
    ) -> None:
        linear_partitions = get_source_partitions(
            gm.graph, [torch.nn.Linear, torch.nn.functional.linear]
        )
        linear_partitions = list(
            itertools.chain.from_iterable(linear_partitions.values())
        )
        for partition in linear_partitions:
            if len(partition.output_nodes) > 1:
                raise ValueError(
                    "Linear partition cannot have more than one output node"
                )
            linear_node = partition.output_nodes[0]
            if linear_node.op != "call_function" or linear_node.target not in (
                torch.ops.aten.linear.default,
            ):
                raise ValueError(f"{linear_node} is not an aten linear operator")
            # skip annotation if it is already annotated
            if skip_annotate([linear_node], filter_fn):
                continue
            self._annotate_linear_node_helper(linear_node, True, quantization_config)

    def _annotate_linear_unary(
        self,
        gm: torch.fx.GraphModule,
        quantization_config: Optional[QuantizationConfig],
        filter_fn: Optional[FilterFn] = None,
    ) -> None:
        postop_list = [
            torch.nn.ReLU,
            torch.nn.LeakyReLU,
            torch.nn.Tanh,
            torch.nn.GELU,
        ]
        fused_partitions: list[tuple] = []
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
                torch.ops.aten.linear.default,
            ):
                continue
            if skip_annotate([unary_node, linear_node], filter_fn):
                continue

            if quantization_config is None:
                annotate_nodes_not_quantize([linear_node, unary_node])
                continue

            self._annotate_linear_node_helper(linear_node, False, quantization_config)
            unary_node.meta[
                QUANT_ANNOTATION_KEY
            ] = OnednnInductorQuantizationAnnotation(
                _annotated=True,
                _is_output_of_quantized_pattern=True,
            )

    def validate(self, model: torch.fx.GraphModule) -> None:
        pass
