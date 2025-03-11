# mypy: allow-untyped-defs
import functools
from typing import Any, Optional, TYPE_CHECKING

import torch
from torch.ao.quantization.observer import HistogramObserver, PerChannelMinMaxObserver
from torch.ao.quantization.quantizer.quantizer import QuantizationSpec
from torch.ao.quantization.quantizer.x86_inductor_quantizer import (
    _is_any_annotated,
    FilterFn,
    int8_in_int8_out_ops,
    X86InductorQuantizer,
)
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import QuantizationConfig
from torch.fx import Node


if TYPE_CHECKING:
    from torch.ao.quantization.qconfig import _ObserverOrFakeQuantizeConstructor

__all__ = [
    "XPUInductorQuantizer",
    "get_default_xpu_inductor_quantization_config",
]


@functools.lru_cache
def get_default_xpu_inductor_quantization_config():
    extra_args: dict[str, Any] = {"eps": 2**-12}
    act_observer_or_fake_quant_ctr = HistogramObserver
    act_quantization_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        qscheme=torch.per_tensor_affine,
        is_dynamic=False,
        observer_or_fake_quant_ctr=act_observer_or_fake_quant_ctr.with_args(
            **extra_args
        ),
    )

    weight_observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor = (
        PerChannelMinMaxObserver
    )

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
        False,
    )
    return quantization_config


class XPUInductorQuantizer(X86InductorQuantizer):
    """
    XPUInductorQuantizer is a class designed to facilitate
    quantization capability at Intel GPU backend. The class
    highly reuses the existing implementation of
    X86InductorQuantizer as both are intended to take advantage
    of the optimized kernels in oneDNN library.
    """

    def __init__(self) -> None:
        super().__init__()

    """
        Following annotate_xx overrides the impls in base class, as
        no XPU implementation for these operators currently. We would
        gradually enable the XPU implementation and remove following
        overrides. We keep the annotate methods but make the function
        body empty, aiming to let `_generate_qdq_quantized_model`
        generate qdq around op and graph execute on fp32 dtype for
        unspported operators.
    """

    def _annotate_qat_conv2d_fusion_pattern(
        self,
        model: torch.fx.GraphModule,
        quantization_config: Optional[QuantizationConfig],
        filter_fn: Optional[FilterFn] = None,
    ):
        pass

    def _annotate_maxpool2d(
        self,
        node: Node,
        quantization_config: Optional[QuantizationConfig],
    ) -> None:
        """
        Here we skip the annotate logic for maxpool at XPU backend
        as the quantized::max_pool2d is only implemented for CPU.
        """
        return

    def _annotate_output_for_int8_in_int8_out_pattern(
        self,
        node: Node,
    ) -> None:
        if (node.target in int8_in_int8_out_ops) and (_is_any_annotated([node])):
            if node.target == torch.ops.aten.max_pool2d.default:
                return
            else:
                input_node = node.all_input_nodes[0]
                self._annotate_output_share_observer_as_input(input_node, node)
        return
