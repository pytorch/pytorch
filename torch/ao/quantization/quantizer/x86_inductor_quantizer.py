# mypy: allow-untyped-defs
import functools
import itertools
import operator
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable, Optional, TYPE_CHECKING
from typing_extensions import TypeAlias

import torch
import torch.nn.functional as F
from torch.ao.quantization.fake_quantize import (
    FakeQuantize,
    FusedMovingAvgObsFakeQuantize,
)
from torch.ao.quantization.observer import (
    HistogramObserver,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    PerChannelMinMaxObserver,
    PlaceholderObserver,
)
from torch.ao.quantization.pt2e.graph_utils import find_sequential_partitions
from torch.ao.quantization.quantizer.quantizer import (
    QuantizationSpec,
    SharedQuantizationSpec,
)
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (
    get_input_act_qspec,
    get_output_act_qspec,
    QuantizationConfig,
)
from torch.fx import Node

from .onednn_inductor_quantizer import (
    _annotate_nodes_not_quantize,
    _create_module_name_filter,
    _create_operator_type_filter,
    _is_all_annotated,
    _is_any_annotated,
    _is_quantized_op_pt2e,
    _mark_nodes_as_annotated,
    _OnednnInductorQuantizationAnnotation,
    _skip_annotate,
    OnednnInductorQuantizer,
    QUANT_ANNOTATION_KEY,
)


FilterFn: TypeAlias = Callable[[list[Node]], bool]


if TYPE_CHECKING:
    from torch.ao.quantization.qconfig import _ObserverOrFakeQuantizeConstructor

__all__ = [
    "X86InductorQuantizer",
    "get_default_x86_inductor_quantization_config",
    "get_x86_inductor_linear_dynamic_fp16_config",
]


@dataclass
class _X86InductorQuantizationAnnotation(_OnednnInductorQuantizationAnnotation):
    # _is_output_of_quantized_pattern:
    #  * Node as output node of a fusion pattern.
    #  * The fusion pattern supports int8 data type.
    #  * The fusion pattern has inputs annotated to insert observer.
    #  * The quantization_config is not `None`.
    pass


# Operators that:
# 1. Operators are optimized to run with int8 when int8 input provided.
# 2. Operators do not support int8 input and produce fp32 output.
int8_in_int8_out_ops: set = {
    torch.ops.aten.max_pool2d.default,
    torch.ops.aten.cat.default,
    torch.ops.aten.avg_pool2d.default,
    torch.ops.aten.adaptive_avg_pool2d.default,
    torch.ops.aten.flatten.using_ints,
}

# Operators that support the int8 data type for quantization config propagation.
# A superset of int8_in_int8_out_ops incorporating additional operators.
propagation_quantizable_ops = int8_in_int8_out_ops

# Operators support the int8 data type
# and recipe is configured by default in X86InductorQuantizer.
default_quantizable_ops = propagation_quantizable_ops | {
    torch.ops.aten.conv2d.default,
    torch.ops.aten.linear.default,
}

# A superset of default_quantizable_ops includes operators support the int8 data type
# but not enabled by default recipe of X86InductorQuantizer.
quantizable_ops = default_quantizable_ops | {
    torch.ops.aten.matmul.default,
}


def _global_config_filter(nodes: list[Node]) -> bool:
    """Filter function for global configuration.

    This filter function takes a list of nodes and returns True if there is exactly one node
    in the list that is a default quantizable operation, False otherwise.
    """
    num_nodes_in_default_quantizable_ops = sum(
        node.target in default_quantizable_ops for node in nodes
    )
    if num_nodes_in_default_quantizable_ops > 1:
        raise NotImplementedError(
            "Several nodes within a single pattern are default quantizable operations."
        )
    return num_nodes_in_default_quantizable_ops == 1


def _map_module_function_to_aten_operator_type():
    module_function_to_aten_operator: dict[Callable, torch._ops.OpOverloadPacket] = {}
    map_list = (
        ([torch.nn.Conv2d, F.conv2d], torch.ops.aten.conv2d.default),
        ([torch.nn.Linear, F.linear], torch.ops.aten.linear.default),
        ([torch.nn.MaxPool2d, F.max_pool2d], torch.ops.aten.max_pool2d.default),
        (
            [
                torch.cat,
            ],
            torch.ops.aten.cat.default,
        ),
        ([torch.nn.AvgPool2d, F.avg_pool2d], torch.ops.aten.avg_pool2d.default),
        (
            [torch.nn.AdaptiveAvgPool2d, F.adaptive_avg_pool2d],
            torch.ops.aten.adaptive_avg_pool2d.default,
        ),
        (
            [
                torch.flatten,
            ],
            torch.ops.aten.flatten.using_ints,
        ),
        (
            [
                torch.matmul,
            ],
            torch.ops.aten.matmul.default,
        ),
    )
    for map_item in map_list:
        module_function_to_aten_operator.update(dict.fromkeys(map_item[0], map_item[1]))  # type: ignore[arg-type, call-overload]
    return module_function_to_aten_operator


@functools.lru_cache
def get_default_x86_inductor_quantization_config(
    is_qat: bool = False,
    is_dynamic: bool = False,
    reduce_range: bool = False,
):
    """
    reduce_range is False by default. Set it to True on earlier CPUs without VNNI to avoid accuracy issue.
    """
    extra_args: dict[str, Any] = {"eps": 2**-12}
    if is_qat:
        if is_dynamic:
            act_observer_or_fake_quant_ctr = FakeQuantize
            dynamic_quant_observer = MovingAverageMinMaxObserver.with_args(
                averaging_constant=1
            )
            extra_args["observer"] = dynamic_quant_observer
        else:
            act_observer_or_fake_quant_ctr = FusedMovingAvgObsFakeQuantize  # type: ignore[assignment]
    else:
        if is_dynamic:
            act_observer_or_fake_quant_ctr = PlaceholderObserver  # type: ignore[assignment]
        else:
            act_observer_or_fake_quant_ctr = HistogramObserver  # type: ignore[assignment]

    # Copy from x86 default qconfig from torch/ao/quantization/qconfig.py
    act_quantization_spec = QuantizationSpec(
        dtype=torch.uint8,
        quant_min=0,
        quant_max=127 if reduce_range else 255,
        qscheme=torch.per_tensor_affine,
        is_dynamic=is_dynamic,
        observer_or_fake_quant_ctr=act_observer_or_fake_quant_ctr.with_args(
            **extra_args
        ),
    )

    weight_observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor = (
        FusedMovingAvgObsFakeQuantize if is_qat else PerChannelMinMaxObserver
    )

    if is_qat:
        # Only support per channel quant for now
        extra_args["observer"] = MovingAveragePerChannelMinMaxObserver  # type: ignore[dict-item]
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
    bias_quantization_spec = None  # will use placeholder observer by default
    quantization_config = QuantizationConfig(
        act_quantization_spec,
        act_quantization_spec,
        weight_quantization_spec,
        bias_quantization_spec,
        is_qat,
    )
    return quantization_config


@functools.lru_cache
def get_x86_inductor_linear_dynamic_fp16_config():
    """
    For linear_dynamic_fp16. The name may be confusing.
    The op's behavior is fp32_input * (fp16_weight -> to_fp32) -> fp32_output.
    """
    weight_quantization_spec = QuantizationSpec(
        dtype=torch.float16,
        observer_or_fake_quant_ctr=PlaceholderObserver,
    )
    quantization_config = QuantizationConfig(
        None,  # input_quantization_spec
        None,  # output_quantization_spec
        weight_quantization_spec,
        None,  # bias_quantization_spec
    )
    return quantization_config


def _config_checker(method: Callable) -> Callable:
    @functools.wraps(method)
    def wrapper(
        quantizer: "X86InductorQuantizer",
        name: Any,
        quantization_config: Optional["QuantizationConfig"],
    ) -> "X86InductorQuantizer":
        if quantizer._need_skip_config(quantization_config):
            warnings.warn(
                f"Skip the quantization config for {name}.",
            )
            return quantizer
        return method(quantizer, name, quantization_config)

    return wrapper


class X86InductorQuantizer(OnednnInductorQuantizer):
    module_function_to_aten_operator_type = _map_module_function_to_aten_operator_type()

    def get_global_quantization_config(self):
        if not isinstance(self.global_config, QuantizationConfig):
            warnings.warn(
                "The global_config for X86InductorQuantizer is currently invalid. \
                Please ensure that you use set_global to establish the global quantization configuration."
            )
        return self.global_config

    @_config_checker
    def set_function_type_qconfig(
        self,
        function_type: Callable,
        quantization_config: Optional[QuantizationConfig],
    ) -> "X86InductorQuantizer":
        if function_type in X86InductorQuantizer.module_function_to_aten_operator_type:
            self._set_aten_operator_qconfig(
                X86InductorQuantizer.module_function_to_aten_operator_type[
                    function_type
                ],
                quantization_config,
            )
        else:
            warnings.warn(
                f"function: Unable to customize quantization config for {function_type} by X86InductorQuantizer."
            )
        return self

    @_config_checker
    def set_module_type_qconfig(
        self,
        module_type: torch.nn.Module,
        quantization_config: Optional[QuantizationConfig],
    ) -> "X86InductorQuantizer":
        if module_type in X86InductorQuantizer.module_function_to_aten_operator_type:
            self._set_aten_operator_qconfig(
                X86InductorQuantizer.module_function_to_aten_operator_type[module_type],
                quantization_config,
            )
        else:
            warnings.warn(
                f"Module: Unable to customize quantization config for {module_type} by X86InductorQuantizer."
            )
        return self

    @_config_checker
    def set_module_name_qconfig(
        self, module_name: str, quantization_config: Optional[QuantizationConfig]
    ):
        """Set quantization_config for a submodule with name: `module_name`, for example:
        quantizer.set_module_name_qconfig("blocks.sub"), it will quantize all supported operator/operator
        patterns in the submodule with this module name with the given `quantization_config`

        The supported operators include `quantizable_ops` and `propagation_quantizable_ops`.
        """
        self.module_name_qconfig[module_name] = quantization_config
        return self

    def _set_aten_operator_qconfig(
        self,
        operator_type: torch._ops.OpOverloadPacket,
        quantization_config: Optional[QuantizationConfig],
    ) -> "X86InductorQuantizer":
        if operator_type in quantizable_ops:
            self.operator_type_qconfig[operator_type] = quantization_config
        else:
            warnings.warn(
                f"operator: Unable to quantize {operator} by X86InductorQuantizer."
            )
        return self

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """Annotate the given model with quantization configurations.

        Annotation contracts:
        1. Annotate each node according to the user's qconfig in the following order:
        `module_name_qconfig`, `operator_type_qconfig`, and `global_config`.
        2. Avoid re-annotating nodes already annotated in prior stages. For example,
        if `linear1` has been annotated by `module_name_qconfig`, it won't be annotated again
        during the processing of the 'operator_type_qconfig' or 'global_config'.
        3. For config is `None`, the node will be annotated with `_X86InductorQuantizationAnnotation(_annotated=True)`.

        For each pair of (module_name_or_operator_type_or_global, qconfig), a filter function is created.
        This filter function checks if the node is marked by current stage and not annotated by the previous stage.
        """
        for module_name, quantization_config in self.module_name_qconfig.items():
            self._annotate_with_config(
                model, quantization_config, _create_module_name_filter(module_name)
            )

        for operator_type, quantization_config in self.operator_type_qconfig.items():
            self._annotate_with_config(
                model, quantization_config, _create_operator_type_filter(operator_type)
            )

        if self.global_config:
            self._annotate_with_config(
                model,
                self.global_config,
                _global_config_filter,
            )

        # Once we've annotated the model with quantization configurations, we also need to annotate
        # the output of quantizable operations. For example, if we annotated `maxpool2d` to quantize its inputs,
        # we will quantize its output accordingly. This enables us to fuse the dq-operator-q into a quantized op.
        # Refer to https://github.com/intel/intel-extension-for-pytorch/blob/
        # 90d19323d96afc53fcc22ba5a7bb3fb07fdd6c1c/intel_extension_for_pytorch/quantization/_recipe.py#L487

        self._annotate_output_for_int8_in_int8_out_pattern_entry(model)

        return model

    def _annotate_with_config(
        self,
        model: torch.fx.GraphModule,
        quantization_config: Optional[QuantizationConfig],
        filter_fn: FilterFn,
    ) -> None:
        """Annotate the model with the given quantization configuration.

        High-level description of quantization recipe for X86 Inductor Backend:
        Step 1: Apply quantization recipe for fusion patterns of conv/linear to enable int8 data type actively.
        Step 2: Propagate quantization annotation for patterns besides conv/linear. Go through the pattern in model
        from start to the end. If a pattern supports computation with int8 data type and inputs connected to
        quantized patterns, annotate its inputs as quantized pattern.
        """

        # Step1: Recipe of fusion patterns like conv/linear.
        self._annotate_conv2d_fusion_pattern(model, quantization_config, filter_fn)
        self._annotate_linear_fusion_pattern(model, quantization_config, filter_fn)
        self._annotate_matmul(model, quantization_config, filter_fn)

        # Step2: Recipe to propagate annotation for patterns beside conv/linear.
        # Go through all the nodes from start to end.
        # Recipe refer to https://github.com/intel/intel-extension-for-pytorch/blob/
        # 90d19323d96afc53fcc22ba5a7bb3fb07fdd6c1c/intel_extension_for_pytorch/quantization/_recipe.py#L538

        self._annotate_propagation_quantizable_pattern_entry(
            model, quantization_config, filter_fn
        )

    def _annotate_qat_conv2d_fusion_pattern(
        self,
        model: torch.fx.GraphModule,
        quantization_config: Optional[QuantizationConfig],
        filter_fn: Optional[FilterFn] = None,
    ):
        # Annotate QAT Specific patterns
        self._annotate_qat_conv2d_bn_binary_unary(model, quantization_config, filter_fn)
        self._annotate_qat_conv2d_bn_binary(model, quantization_config, filter_fn)
        self._annotate_qat_conv2d_bn_unary(model, quantization_config, filter_fn)
        self._annotate_qat_conv2d_bn(model, quantization_config, filter_fn)

    def _annotate_qat_conv2d_bn_binary_unary(
        self,
        gm: torch.fx.GraphModule,
        quantization_config: Optional[QuantizationConfig],
        filter_fn: Optional[FilterFn] = None,
    ) -> None:
        fused_partitions = find_sequential_partitions(
            gm, [torch.nn.Conv2d, torch.nn.BatchNorm2d, operator.add, torch.nn.ReLU]
        )
        for fused_partition in fused_partitions:
            (
                conv_partition,
                bn_partition,
                binary_partition,
                unary_partition,
            ) = fused_partition

            (
                conv_node,
                bn_output_node,
                binary_node,
                unary_node,
            ) = self._get_output_nodes_of_partitions(
                [conv_partition, bn_partition, binary_partition, unary_partition]
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

            if _skip_annotate(
                [unary_node, binary_node, bn_output_node, conv_node], filter_fn
            ):
                continue

            self._annotate_conv_node_helper(conv_node, False, quantization_config)

            if quantization_config is not None:
                binary_node_input_qspec_map = {}
                binary_node_input_qspec_map[extra_input_node] = get_input_act_qspec(
                    quantization_config
                )
                binary_node.meta[
                    QUANT_ANNOTATION_KEY
                ] = _X86InductorQuantizationAnnotation(
                    input_qspec_map=binary_node_input_qspec_map,
                    _annotated=True,
                )
                unary_node.meta[
                    QUANT_ANNOTATION_KEY
                ] = _X86InductorQuantizationAnnotation(
                    # TODO<leslie> Remove the annotate of output in QAT when qat util support pattern matcher.
                    output_qspec=get_output_act_qspec(quantization_config),  # type: ignore[arg-type]
                    _annotated=True,
                    _is_output_of_quantized_pattern=True,
                )
            else:
                _annotate_nodes_not_quantize([binary_node, unary_node])
            nodes_to_mark_annotated = list(conv_partition.nodes)
            nodes_to_mark_annotated.extend(list(bn_partition.nodes))
            nodes_to_mark_annotated.extend(list(binary_partition.nodes))
            nodes_to_mark_annotated.extend(list(unary_partition.nodes))
            _mark_nodes_as_annotated(nodes_to_mark_annotated)

    def _annotate_qat_conv2d_bn_unary(
        self,
        gm: torch.fx.GraphModule,
        quantization_config: Optional[QuantizationConfig],
        filter_fn: Optional[FilterFn] = None,
    ) -> None:
        fused_partitions = []
        unary_patterns = [
            [torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU],
            [torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.Hardtanh],
            [torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.Hardswish],
            [torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU6],
            [torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.SiLU],
        ]
        for unary_pattern in unary_patterns:
            partitions = find_sequential_partitions(gm, unary_pattern)
            if partitions:
                # Extend the fused_partitions if partitions is not empty
                fused_partitions.extend(partitions)

        for fused_partition in fused_partitions:
            conv_partition, bn_partition, unary_partition = fused_partition
            (
                conv_node,
                bn_output_node,
                unary_node,
            ) = self._get_output_nodes_of_partitions(
                [conv_partition, bn_partition, unary_partition]
            )

            if (
                conv_node.op != "call_function"
                or conv_node.target != torch.ops.aten.conv2d.default
            ):
                continue

            if _skip_annotate([unary_node, bn_output_node, conv_node], filter_fn):
                continue

            self._annotate_conv_node_helper(conv_node, False, quantization_config)
            if quantization_config is not None:
                unary_node.meta[
                    QUANT_ANNOTATION_KEY
                ] = _X86InductorQuantizationAnnotation(
                    # TODO<leslie> Remove the annotate of output in QAT when qat util support pattern matcher.
                    output_qspec=get_output_act_qspec(quantization_config),  # type: ignore[arg-type]
                    _annotated=True,
                    _is_output_of_quantized_pattern=True,
                )
            else:
                _annotate_nodes_not_quantize(unary_node)
            nodes_to_mark_annotated = list(conv_partition.nodes)
            nodes_to_mark_annotated.extend(list(bn_partition.nodes))
            nodes_to_mark_annotated.extend(list(unary_partition.nodes))
            _mark_nodes_as_annotated(nodes_to_mark_annotated)

    def _annotate_conv2d_fusion_pattern(
        self,
        model: torch.fx.GraphModule,
        quantization_config: Optional[QuantizationConfig],
        filter_fn: Optional[FilterFn] = None,
    ):
        if (quantization_config is None) or (quantization_config.is_qat):
            # Annotate QAT specific pattern: mainly due to BN not folded in prepare_qat
            self._annotate_qat_conv2d_fusion_pattern(
                model, quantization_config, filter_fn
            )
        self._annotate_conv2d_binary_unary(model, quantization_config, filter_fn)
        self._annotate_conv2d_binary(model, quantization_config, filter_fn)
        self._annotate_conv2d_unary(model, quantization_config, filter_fn)
        self._annotate_conv2d(model, quantization_config, filter_fn)

    def _annotate_linear_fusion_pattern(
        self,
        model: torch.fx.GraphModule,
        quantization_config: Optional[QuantizationConfig],
        filter_fn: Optional[FilterFn] = None,
    ):
        self._annotate_linear_binary_unary(model, quantization_config, filter_fn)
        self._annotate_linear_unary(model, quantization_config, filter_fn)
        self._annotate_linear(model, quantization_config, filter_fn)

    def _annotate_conv2d_binary_unary(
        self,
        gm: torch.fx.GraphModule,
        quantization_config: Optional[QuantizationConfig],
        filter_fn: Optional[FilterFn] = None,
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
            if (
                conv_node.op != "call_function"
                or conv_node.target != torch.ops.aten.conv2d.default
            ):
                # No conv node found to be fused with add
                continue
            if _skip_annotate([unary_node, binary_node, conv_node], filter_fn):
                continue

            if quantization_config is None:
                _annotate_nodes_not_quantize([conv_node, binary_node, unary_node])
                continue

            self._annotate_conv_node_helper(conv_node, False, quantization_config)
            binary_node_input_qspec_map = {}
            binary_node_input_qspec_map[extra_input_node] = get_input_act_qspec(
                quantization_config
            )
            binary_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(
                input_qspec_map=binary_node_input_qspec_map,
                _annotated=True,
            )
            unary_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(
                _annotated=True,
                _is_output_of_quantized_pattern=True,
            )

    def _annotate_conv2d_unary(
        self,
        gm: torch.fx.GraphModule,
        quantization_config: Optional[QuantizationConfig],
        filter_fn: Optional[FilterFn] = None,
    ) -> None:
        fused_partitions = []
        unary_patterns = [
            [torch.nn.Conv2d, torch.nn.ReLU],
            [torch.nn.Conv2d, torch.nn.Hardtanh],
            [torch.nn.Conv2d, torch.nn.Hardswish],
            [torch.nn.Conv2d, torch.nn.ReLU6],
            [torch.nn.Conv2d, torch.nn.SiLU],
        ]
        for unary_pattern in unary_patterns:
            partitions = find_sequential_partitions(gm, unary_pattern)
            if partitions:
                # Extend the fused_partitions if partitions is not empty
                fused_partitions.extend(partitions)

        for fused_partition in fused_partitions:
            conv_partition, unary_partition = fused_partition
            conv_node, unary_node = self._get_output_nodes_of_partitions(
                [conv_partition, unary_partition]
            )
            if (
                conv_node.op != "call_function"
                or conv_node.target != torch.ops.aten.conv2d.default
            ):
                continue
            if _skip_annotate([unary_node, conv_node], filter_fn):
                continue

            if quantization_config is None:
                _annotate_nodes_not_quantize([conv_node, unary_node])
                continue

            self._annotate_conv_node_helper(conv_node, False, quantization_config)
            unary_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(
                _annotated=True,
                _is_output_of_quantized_pattern=True,
            )

    def _annotate_maxpool2d(
        self,
        node: Node,
        quantization_config: Optional[QuantizationConfig],
    ) -> None:
        if node.target is not torch.ops.aten.max_pool2d.default:
            return
        if quantization_config is None:
            _annotate_nodes_not_quantize(node)
            return

        maxpool_node = node
        if _is_any_annotated(
            [
                maxpool_node,
            ]
        ):
            return

        input_node = maxpool_node.args[0]
        assert isinstance(input_node, Node)
        input_qspec_map = {}
        input_qspec_map[input_node] = get_input_act_qspec(quantization_config)
        maxpool_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            _annotated=True,
            _is_output_of_quantized_pattern=True,
        )

    def _annotate_cat(
        self, node: Node, quantization_config: QuantizationConfig
    ) -> None:
        if quantization_config is None:
            _annotate_nodes_not_quantize(node)
            return
        cat_node = node
        input_nodes = cat_node.args[0]
        assert isinstance(input_nodes, Sequence)
        first_input_node = input_nodes[0]
        input_qspec_map = {}
        assert isinstance(first_input_node, Node)
        assert isinstance(cat_node, Node)
        input_qspec_map[first_input_node] = get_input_act_qspec(quantization_config)
        share_qparams_with_input_act0_qspec = SharedQuantizationSpec(
            (first_input_node, cat_node)
        )

        for input_node in input_nodes[1:]:
            if input_node not in input_qspec_map:
                # There has the case of cat same nodes: torch.cat([input0, input0], 1)
                assert isinstance(input_node, Node)
                input_qspec_map[input_node] = share_qparams_with_input_act0_qspec

        cat_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            _annotated=True,
            _is_output_of_quantized_pattern=True,
        )

    def _annotate_propagation_quantizable_pattern_entry(
        self,
        gm: torch.fx.GraphModule,
        quantization_config: Optional[QuantizationConfig],
        filter_fn: Optional[FilterFn] = None,
    ):
        for node in gm.graph.nodes:
            self._annotate_propagation_quantizable_pattern(
                node, quantization_config, filter_fn
            )

    def _annotate_propagation_quantizable_pattern(
        self, node: Node, quantization_config, filter_fn
    ) -> None:
        # Propagate annotation to quantizable patterns.
        if (
            (node.target in propagation_quantizable_ops)
            and (not _is_any_annotated([node]))
            and (node.op == "call_function")
        ):

            def is_all_inputs_connected_to_quantized_op(input_nodes):
                # Ensure all the inputs connect to fusion pattern or quantized node
                for input_node in input_nodes:
                    if not _is_quantized_op_pt2e(input_node):
                        return False
                return True

            if _skip_annotate([node], filter_fn):
                return

            if quantization_config is None:
                _annotate_nodes_not_quantize(node)
                return

            if node.target is torch.ops.aten.max_pool2d.default:
                # Recipe of maxpool2d: check input arg[0] of maxpool2d is quantized or not
                input_nodes_to_check = [node.all_input_nodes[0]]
                if not is_all_inputs_connected_to_quantized_op(input_nodes_to_check):
                    if quantization_config is not None:
                        warnings.warn(
                            f"The input of maxpool2d is not quantized, skip annotate maxpool2d with config {quantization_config}."
                        )
                    return

                self._annotate_maxpool2d(node, quantization_config)
                return
            elif node.target is torch.ops.aten.cat.default:
                input_nodes_to_check = node.all_input_nodes
                if not is_all_inputs_connected_to_quantized_op(input_nodes_to_check):
                    return
                self._annotate_cat(node, quantization_config)
            elif (
                node.target is torch.ops.aten.flatten.using_ints
                and len(node.users) > 0
                and not any(
                    user.target in quantizable_ops for user in node.users.keys()
                )
            ):
                # Recipe of flatten: check if any users of flatten node are quantizable ops or not
                return
            else:
                input_node = node.all_input_nodes[0]
                if not is_all_inputs_connected_to_quantized_op(
                    [
                        input_node,
                    ]
                ):
                    return
                input_qspec_map = {}
                input_qspec_map[input_node] = get_input_act_qspec(quantization_config)
                node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(
                    input_qspec_map=input_qspec_map,
                    _annotated=True,
                    _is_output_of_quantized_pattern=True,
                )
        return

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

    def _annotate_output_for_int8_in_int8_out_pattern_entry(
        self,
        model: torch.fx.GraphModule,
    ):
        for node in model.graph.nodes:
            self._annotate_output_for_int8_in_int8_out_pattern(node)

    def _annotate_output_for_int8_in_int8_out_pattern(
        self,
        node: Node,
    ) -> None:
        r"""
        Check and insert observer at output of node in int8_in_int8_out_ops if needed.
        Recipe refers to https://github.com/intel/intel-extension-for-pytorch/blob/
        90d19323d96afc53fcc22ba5a7bb3fb07fdd6c1c/intel_extension_for_pytorch/quantization/_utils.py#L495
        """
        edge_or_node: tuple[Node, Node]
        if (node.target in int8_in_int8_out_ops) and (_is_any_annotated([node])):
            if node.target == torch.ops.aten.max_pool2d.default:
                maxpool_node = node
                if not _is_all_annotated(
                    [
                        maxpool_node,
                    ]
                ):
                    return

                # Get the quantization_annotation from getitem_node
                maxpool_node_quantization_annotation = (
                    maxpool_node.meta[QUANT_ANNOTATION_KEY]
                    if QUANT_ANNOTATION_KEY in maxpool_node.meta
                    else None
                )
                if (
                    maxpool_node_quantization_annotation
                    and maxpool_node_quantization_annotation._is_output_of_quantized_pattern
                ):
                    # Annotate the output_qspec of getitem_node
                    input_act = maxpool_node.args[0]
                    assert isinstance(input_act, Node)
                    assert isinstance(maxpool_node, Node)
                    edge_or_node = (input_act, maxpool_node)
                    maxpool_node_quantization_annotation.output_qspec = (
                        SharedQuantizationSpec(edge_or_node)
                    )
            else:
                input_node = node.all_input_nodes[0]
                self._annotate_output_share_observer_as_input(input_node, node)
        return

    def _annotate_linear_binary_unary(
        self,
        gm: torch.fx.GraphModule,
        quantization_config: Optional[QuantizationConfig],
        filter_fn: Optional[FilterFn] = None,
    ) -> None:
        # linear + binary_op + (optional) unary op
        binary_op_list = [operator.add]
        unary_op_list = [torch.nn.ReLU, None]
        combinations = itertools.product(binary_op_list, unary_op_list)
        for binary_op, unary_op in combinations:
            has_unary = unary_op is not None
            seq_partition = [torch.nn.Linear, binary_op]
            if has_unary:
                seq_partition.append(unary_op)
            fused_partitions = find_sequential_partitions(gm, seq_partition)
            for fused_partition in fused_partitions:
                unary_partition, unary_node = None, None
                if has_unary:
                    (
                        linear_partition,
                        binary_partition,
                        unary_partition,
                    ) = fused_partition
                    (
                        linear_node,
                        binary_node,
                        unary_node,
                    ) = self._get_output_nodes_of_partitions(
                        [linear_partition, binary_partition, unary_partition]
                    )
                else:
                    linear_partition, binary_partition = fused_partition
                    linear_node, binary_node = self._get_output_nodes_of_partitions(
                        [linear_partition, binary_partition]
                    )
                if len(linear_node.users) != 1:
                    # Linear Node should only has 1 user node
                    continue
                (
                    linear_node_idx,
                    extra_input_node_idx,
                ) = self._get_input_idx_for_binary_node(linear_node, binary_node)
                if (linear_node_idx is None) or (extra_input_node_idx is None):
                    continue
                if linear_node != binary_node.args[linear_node_idx]:
                    raise ValueError(
                        f"{linear_node} doesn't match input of binary node"
                    )
                assert isinstance(linear_node, Node)
                if (
                    linear_node.op != "call_function"
                    or linear_node.target != torch.ops.aten.linear.default
                ):
                    # No linear node found to be fused with add
                    continue
                node_list = (
                    [binary_node, linear_node]
                    if unary_node is None
                    else [unary_node, binary_node, linear_node]
                )
                if _skip_annotate(node_list, filter_fn):
                    continue

                if quantization_config is None:
                    _annotate_nodes_not_quantize(node_list)
                    continue

                self._annotate_linear_node_helper(
                    linear_node, False, quantization_config
                )
                # We don't insert q-dq before the binary input node due to accuracy issues
                binary_node.meta[
                    QUANT_ANNOTATION_KEY
                ] = _X86InductorQuantizationAnnotation(
                    input_qspec_map={},
                    _annotated=True,
                    _is_output_of_quantized_pattern=(not has_unary),
                )
                if unary_node is not None:
                    unary_node.meta[
                        QUANT_ANNOTATION_KEY
                    ] = _X86InductorQuantizationAnnotation(
                        _annotated=True,
                        _is_output_of_quantized_pattern=True,
                    )
