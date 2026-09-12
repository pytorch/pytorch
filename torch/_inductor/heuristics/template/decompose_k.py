from __future__ import annotations

from typing import Any, TYPE_CHECKING

import sympy

from torch._inductor import config
from torch._inductor.heuristics.registry import register_template_heuristic
from torch.utils._ordered_set import OrderedSet

from ...ir import get_free_symbols
from ...kernel.decompose_k import (
    BLACKWELL_DECOMPOSE_K_PARTIAL_CONFIGS,
    decompose_k_subgraph_template,
    get_blackwell_decompose_k_splits,
)
from ...kernel_inputs import KernelInputs, MMKernelInputs
from ...runtime.hints import DeviceProperties
from ...utils import get_k_splits, use_triton_blackwell_tma_template
from ...virtualized import V
from .base import TemplateConfigHeuristics
from .gemm import GemmMaxAutotuneTemplateConfigHeuristics


if TYPE_CHECKING:
    from collections.abc import Generator


@register_template_heuristic(decompose_k_subgraph_template.uid, None, op_name="mm")
class EmptyDecomposeKConfigHeuristics(TemplateConfigHeuristics):
    """empty heuristics to skip decompose k on anything not cuda"""


@register_template_heuristic(
    decompose_k_subgraph_template.uid,
    "xpu",
    op_name="mm",
)
# Register on CUDA (both NVIDIA and ROCm/HIP)
# Runtime enablement is controlled by config.triton.num_decompose_k_splits (0 disables)
@register_template_heuristic(
    decompose_k_subgraph_template.uid,
    "cuda",
    op_name="mm",
)
# TODO(coconutruben): enable decompose k on other devices (xpu, cpu, mps, mtia)
# by either adding specific register_template_heuristic tags, or setting the
# device to None (enabled on all devices)
class DecomposeKConfigHeuristics(GemmMaxAutotuneTemplateConfigHeuristics):
    """Generate backend-specific decompose-K partial-BMM configurations."""

    def _get_template_configs_impl(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Get all the valid k_splits for the given m, n, k.
        """
        if not isinstance(kernel_inputs, MMKernelInputs):
            raise AssertionError(f"{self.__class__.__name__} requires MMKernelInputs")

        # Check for unbacked symbols - if found, yield nothing
        unbacked_symbols = any(
            len(get_free_symbols(itr, unbacked_only=True)) > 0
            for itr in (
                *kernel_inputs.shapes_symbolic(),
                *kernel_inputs.strides_symbolic(),
            )
        )
        if unbacked_symbols:
            return

        m, n, k = kernel_inputs.mnk_symbolic()
        bmm_backends = OrderedSet(
            backend.strip().upper()
            for backend in config.triton.decompose_k_bmm_backends.split(",")
        )
        k_splits = get_k_splits(m, n, k)
        exact_k_splits = [
            k_split
            for k_split in k_splits
            if V.graph.sizevars.statically_known_true(
                sympy.Eq(sympy.Mod(k, k_split), 0)
            )
        ]

        if "ATEN" in bmm_backends:
            for k_split in exact_k_splits:
                yield {"k_split": k_split, "bmm_backend": "aten"}

        if "TRITON" not in bmm_backends:
            return

        mat1, mat2 = kernel_inputs.mat1mat2()
        layout = kernel_inputs.output_layout()
        if not (
            config.triton.enable_blackwell_decompose_k
            and use_triton_blackwell_tma_template(
                mat1,
                mat2,
                output_layout=layout,
                add_guards=True,
            )
        ):
            return

        # The partial template uses compile-time descriptor geometry, so backed
        # dynamic dimensions must be specialized explicitly. Unbacked symbols
        # were rejected above.
        m_hint, n_hint, k_hint = V.graph.sizevars.guard_int_seq((m, n, k))
        device_properties = DeviceProperties.create(kernel_inputs.device())
        # Keep the Triton search to one M/N-specific schedule family and its
        # one- and two-wave splits. The M=128, N=256 producer-fusion target uses
        # a deeper 1CTA BK64 pipeline; other one-M-tile and narrow-N cases use
        # 1CTA BK128, while wider outputs with at least two M tiles use 2CTA
        # BN256. The whole-plan autotuner retains direct and exact ATen fallbacks.
        has_b_producer = mat2.get_name() not in V.graph.graph_inputs
        if m_hint == 128 and n_hint == 256 and has_b_producer:
            config_index = 6
        else:
            config_index = 2 if m_hint > 128 and n_hint > 128 else 0
        partial_config = BLACKWELL_DECOMPOSE_K_PARTIAL_CONFIGS[config_index]
        for k_split in get_blackwell_decompose_k_splits(
            m_hint,
            n_hint,
            k_hint,
            device_properties.multi_processor_count,
            partial_config,
        ):
            yield {
                "k_split": k_split,
                "bmm_backend": "triton",
                "bmm_config_index": config_index,
            }
