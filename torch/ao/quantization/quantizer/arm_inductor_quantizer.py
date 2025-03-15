# mypy: allow-untyped-defs
import functools
import operator
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING
from typing_extensions import TypeAlias

import torch
import torch.nn.functional as F
from torch.ao.quantization.fake_quantize import (
    FakeQuantize,
    FusedMovingAvgObsFakeQuantize,
)
from torch.ao.quantization.observer import (
    HistogramObserver,
    MinMaxObserver,
    MovingAverageMinMaxObserver,
    PlaceholderObserver,
)
from torch.ao.quantization.quantizer.quantizer import QuantizationSpec
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import QuantizationConfig
from torch.fx import Node

from .onednn_inductor_quantizer import (
    create_module_name_filter,
    create_operator_type_filter,
    OnednnInductorQuantizationAnnotation,
    OnednnInductorQuantizer,
)


FilterFn: TypeAlias = Callable[[List[Node]], bool]


if TYPE_CHECKING:
    from torch.ao.quantization.qconfig import _ObserverOrFakeQuantizeConstructor

__all__ = [
    "ArmInductorQuantizer",
    "get_default_arm_inductor_quantization_config",
]


@dataclass
class _ArmInductorQuantizationAnnotation(OnednnInductorQuantizationAnnotation):
    # _is_output_of_quantized_pattern:
    #  * Node as output node of a fusion pattern.
    #  * The fusion pattern supports int8 data type.
    #  * The fusion pattern has inputs annotated to insert observer.
    #  * The quantization_config is not `None`.
    pass


# Operators support the int8 data type
# and recipe is configured by default in ArmInductorQuantizer.
default_quantizable_ops = {
    torch.ops.aten.conv2d.default,
    torch.ops.aten.linear.default,
}

# A superset of default_quantizable_ops includes operators support the int8 data type
# but not enabled by default recipe of ArmInductorQuantizer.
quantizable_ops = default_quantizable_ops | {
    torch.ops.aten.matmul.default,
}


def _global_config_filter(nodes: List[Node]) -> bool:
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
    module_function_to_aten_operator: Dict[Callable, torch._ops.OpOverloadPacket] = {}
    map_list = (
        ([torch.nn.Conv2d, F.conv2d], torch.ops.aten.conv2d.default),
        ([torch.nn.Linear, F.linear], torch.ops.aten.linear.default),
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
def get_default_arm_inductor_quantization_config(
    is_qat: bool = False,
    is_dynamic: bool = False,
):
    extra_args: Dict[str, Any] = {"eps": 2**-12}
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
    # check for the qconfig -------------------------
    act_quantization_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        qscheme=torch.per_tensor_affine,
        is_dynamic=is_dynamic,
        observer_or_fake_quant_ctr=act_observer_or_fake_quant_ctr.with_args(
            **extra_args
        ),
    )

    weight_observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor = (
        FusedMovingAvgObsFakeQuantize if is_qat else MinMaxObserver
    )

    if is_qat:
        # Only support per tensor quant for now
        extra_args["observer"] = MovingAverageMinMaxObserver  # type: ignore[dict-item]
    weight_quantization_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        qscheme=torch.per_tensor_symmetric,
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


def _config_checker(method: Callable) -> Callable:
    @functools.wraps(method)
    def wrapper(
        quantizer: "ArmInductorQuantizer",
        name: Any,
        quantization_config: Optional["QuantizationConfig"],
    ) -> "ArmInductorQuantizer":
        if quantizer._need_skip_config(quantization_config):
            warnings.warn(
                f"Skip the quantization config for {name}.",
            )
            return quantizer
        return method(quantizer, name, quantization_config)

    return wrapper


class ArmInductorQuantizer(OnednnInductorQuantizer):
    module_function_to_aten_operator_type = _map_module_function_to_aten_operator_type()

    def get_global_quantization_config(self):
        if not isinstance(self.global_config, QuantizationConfig):
            warnings.warn(
                "The global_config for ArmInductorQuantizer is currently invalid. \
                Please ensure that you use set_global to establish the global quantization configuration."
            )
        return self.global_config

    @_config_checker
    def set_function_type_qconfig(
        self,
        function_type: Callable,
        quantization_config: Optional[QuantizationConfig],
    ) -> "ArmInductorQuantizer":
        if function_type in ArmInductorQuantizer.module_function_to_aten_operator_type:
            self._set_aten_operator_qconfig(
                ArmInductorQuantizer.module_function_to_aten_operator_type[
                    function_type
                ],
                quantization_config,
            )
        else:
            warnings.warn(
                f"function: Unable to customize quantization config for {function_type} by ArmInductorQuantizer."
            )
        return self

    @_config_checker
    def set_module_type_qconfig(
        self,
        module_type: torch.nn.Module,
        quantization_config: Optional[QuantizationConfig],
    ) -> "ArmInductorQuantizer":
        if module_type in ArmInductorQuantizer.module_function_to_aten_operator_type:
            self._set_aten_operator_qconfig(
                ArmInductorQuantizer.module_function_to_aten_operator_type[module_type],
                quantization_config,
            )
        else:
            warnings.warn(
                f"Module: Unable to customize quantization config for {module_type} by ArmInductorQuantizer."
            )
        return self

    @_config_checker
    def set_module_name_qconfig(
        self, module_name: str, quantization_config: Optional[QuantizationConfig]
    ):
        """Set quantization_config for a submodule with name: `module_name`, for example:
        quantizer.set_module_name_qconfig("blocks.sub"), it will quantize all supported operator/operator
        patterns in the submodule with this module name with the given `quantization_config`

        The supported operators include `quantizable_ops` only.
        """
        self.module_name_qconfig[module_name] = quantization_config
        return self

    def _set_aten_operator_qconfig(
        self,
        operator_type: torch._ops.OpOverloadPacket,
        quantization_config: Optional[QuantizationConfig],
    ) -> "ArmInductorQuantizer":
        if operator_type in quantizable_ops:
            self.operator_type_qconfig[operator_type] = quantization_config
        else:
            warnings.warn(
                f"operator: Unable to quantize {operator} by ArmInductorQuantizer."
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
        3. For config is `None`, the node will be annotated with `_ArmInductorQuantizationAnnotation(_annotated=True)`.

        For each pair of (module_name_or_operator_type_or_global, qconfig), a filter function is created.
        This filter function checks if the node is marked by current stage and not annotated by the previous stage.
        """
        for module_name, quantization_config in self.module_name_qconfig.items():
            self._annotate_with_config(
                model, quantization_config, create_module_name_filter(module_name)
            )

        for operator_type, quantization_config in self.operator_type_qconfig.items():
            self._annotate_with_config(
                model, quantization_config, create_operator_type_filter(operator_type)
            )

        if self.global_config:
            self._annotate_with_config(
                model,
                self.global_config,
                _global_config_filter,
            )

        return model

    def _annotate_with_config(
        self,
        model: torch.fx.GraphModule,
        quantization_config: Optional[QuantizationConfig],
        filter_fn: FilterFn,
    ) -> None:
        """Annotate the model with the given quantization configuration.

        High-level description of quantization recipe for Arm Inductor Backend:
        Apply quantization recipe for fusion patterns of conv/linear to enable int8 data type actively.
        """

        # Step1: Recipe of fusion patterns like conv/linear.
        self._annotate_conv2d_fusion_pattern(model, quantization_config, filter_fn)
        self._annotate_linear_fusion_pattern(model, quantization_config, filter_fn)
        self._annotate_matmul(model, quantization_config, filter_fn)

    def _annotate_qat_conv2d_fusion_pattern(
        self,
        model: torch.fx.GraphModule,
        quantization_config: Optional[QuantizationConfig],
        filter_fn: Optional[FilterFn] = None,
    ):
        # Annotate QAT Specific patterns
        self._annotate_qat_conv2d_bn_binary(model, quantization_config, filter_fn)
        self._annotate_qat_conv2d_bn(model, quantization_config, filter_fn)

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
        self._annotate_conv2d_binary(model, quantization_config, filter_fn)
        self._annotate_conv2d(model, quantization_config, filter_fn)

    def _annotate_linear_fusion_pattern(
        self,
        model: torch.fx.GraphModule,
        quantization_config: Optional[QuantizationConfig],
        filter_fn: Optional[FilterFn] = None,
    ):
        self._annotate_linear_unary(model, quantization_config, filter_fn)
        self._annotate_linear(model, quantization_config, filter_fn)
