from typing import Optional

import torch
from torch.ao.quantization.quantizer.x86_inductor_quantizer import (
    FilterFn,
    X86InductorQuantizer,
)
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import QuantizationConfig


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

    def _annotate_conv2d_binary(
        self,
        gm: torch.fx.GraphModule,
        quantization_config: Optional[QuantizationConfig],
        filter_fn: Optional[FilterFn] = None,
    ) -> None:
        pass

    def _annotate_conv2d_binary_unary(
        self,
        gm: torch.fx.GraphModule,
        quantization_config: Optional[QuantizationConfig],
        filter_fn: Optional[FilterFn] = None,
    ) -> None:
        pass
