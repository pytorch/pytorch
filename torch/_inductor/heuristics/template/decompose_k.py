from __future__ import annotations

import math
from typing import Any, TYPE_CHECKING

import sympy

import torch
from torch._inductor import config
from torch._inductor.heuristics.registry import register_template_heuristic

from ...ir import get_free_symbols
from ...kernel.decompose_k import (
    BLACKWELL_DECOMPOSE_K_PARTIAL_CONFIGS,
    blackwell_decompose_k_partial_template,
    blackwell_decompose_k_subgraph_template,
    BlackwellDecomposeKPartialKernelInputs,
)
from ...kernel.mm import decompose_k_subgraph_template
from ...kernel_inputs import KernelInputs, MMKernelInputs
from ...runtime.hints import DeviceProperties
from ...utils import get_k_splits, get_num_sms
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
        k_splits = get_k_splits(m, n, k)
        for k_split in k_splits:
            if not V.graph.sizevars.statically_known_true(
                sympy.Eq(sympy.Mod(k, k_split), 0)
            ):
                continue
            yield {"k_split": k_split}


@register_template_heuristic(
    blackwell_decompose_k_partial_template.uid,
    None,
    op_name="blackwell_decompose_k_partial",
)
class EmptyBlackwellDecomposeKPartialConfigHeuristics(TemplateConfigHeuristics):
    """Skip the Blackwell partial template outside NVIDIA CUDA."""


@register_template_heuristic(
    blackwell_decompose_k_partial_template.uid,
    "cuda",
    op_name="blackwell_decompose_k_partial",
)
class BlackwellDecomposeKPartialConfigHeuristics(TemplateConfigHeuristics):
    """Materialize the partial-GEMM config selected by the complete plan."""

    def _get_template_configs_impl(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> Generator[dict[str, Any], None, None]:
        if not isinstance(kernel_inputs, BlackwellDecomposeKPartialKernelInputs):
            raise AssertionError(
                f"{self.__class__.__name__} requires "
                "BlackwellDecomposeKPartialKernelInputs"
            )

        mat1, mat2 = kernel_inputs.mat1mat2()
        m, k = map(int, mat1.get_size())
        k_b, n = map(int, mat2.get_size())
        if k != k_b:
            raise AssertionError(f"incompatible K dimensions: {k} and {k_b}")

        k_split = int(kernel_inputs.get_scalar("k_split"))
        config_index = int(kernel_inputs.get_scalar("config_index"))
        m_pad = int(kernel_inputs.get_scalar("m_pad"))
        k_part = int(kernel_inputs.get_scalar("k_part"))
        partial_config = BLACKWELL_DECOMPOSE_K_PARTIAL_CONFIGS[config_index]
        layout = kernel_inputs.output_layout()
        if layout.dtype != torch.float32:
            raise AssertionError("decompose-K partial workspace must be FP32")
        if list(map(int, layout.size)) != [k_split * m_pad, n]:
            raise AssertionError("invalid decompose-K partial workspace layout")
        if (k_split - 1) * k_part >= k:
            raise NotImplementedError("aligned split leaves an empty final partition")

        m_tiles = m_pad // partial_config.block_m
        template_kwargs = {
            "BLOCK_M": partial_config.block_m,
            "BLOCK_N": partial_config.block_n,
            "BLOCK_K": partial_config.block_k,
            "K_TILES": k_part // partial_config.block_k,
            "GROUP_M": 8,
            "BATCH_SIZE": k_split,
            "LOGICAL_M": m,
            "LOGICAL_N": n,
            "DESCRIPTOR_K": k,
            "A_BATCH_STRIDE": 0,
            "B_BATCH_STRIDE": 0,
            "K_BATCH_OFFSET": k_part,
            "A_M_STRIDE": int(mat1.get_stride()[0]),
            "A_K_STRIDE": int(mat1.get_stride()[1]),
            "B_K_STRIDE": int(mat2.get_stride()[0]),
            "B_N_STRIDE": int(mat2.get_stride()[1]),
            "OUTPUT_BATCH_ROWS": m_pad,
            "NUM_SMS": min(
                get_num_sms(), m_tiles * math.ceil(n / partial_config.block_n)
            ),
            "A_ROW_MAJOR": mat1.get_stride()[1] == 1,
            "B_ROW_MAJOR": mat2.get_stride()[1] == 1,
            "ALLOW_TF32": False,
            "USE_META_WS": True,
            "WARP_SPECIALIZE": True,
            "FLATTEN": False,
            "DATA_PARTITION_FACTOR": partial_config.data_partition_factor,
            "SEPARATE_EPILOGUE_STORE": True,
            "EPILOGUE_SUBTILE": partial_config.epilogue_subtile,
            "TWO_CTAS": partial_config.two_ctas,
            "FLATTEN_OUTPUT": True,
            "tma_store": True,
            "transpose_discontiguous_tensor_descriptors_override": True,
            "num_stages": partial_config.num_stages,
            "num_warps": 8,
        }
        if partial_config.two_ctas:
            template_kwargs["ctas_per_cga"] = (2, 1, 1)
        yield template_kwargs


@register_template_heuristic(
    blackwell_decompose_k_subgraph_template.uid,
    None,
    op_name="mm",
)
class EmptyBlackwellDecomposeKConfigHeuristics(TemplateConfigHeuristics):
    """Skip the experimental partial template outside NVIDIA CUDA."""


@register_template_heuristic(
    blackwell_decompose_k_subgraph_template.uid,
    "cuda",
    op_name="mm",
)
class BlackwellDecomposeKConfigHeuristics(TemplateConfigHeuristics):
    """Generate one bounded 1CTA and one bounded 2CTA complete-plan seed."""

    def _get_template_configs_impl(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> Generator[dict[str, Any], None, None]:
        if not config.triton.enable_blackwell_decompose_k_partial:
            return
        if not isinstance(kernel_inputs, MMKernelInputs):
            raise AssertionError(f"{self.__class__.__name__} requires MMKernelInputs")
        if any(
            len(get_free_symbols(value, unbacked_only=True)) > 0
            for value in (
                *kernel_inputs.shapes_symbolic(),
                *kernel_inputs.strides_symbolic(),
            )
        ):
            return

        device = DeviceProperties.create(kernel_inputs.device())
        if device.type != "cuda" or device.major != 10:
            return
        m_sym, n_sym, k_sym = kernel_inputs.mnk_symbolic()
        m, n, k = int(m_sym), int(n_sym), int(k_sym)

        config_indices = [0]
        if m > 128:
            config_indices.append(1 if n <= 128 else 2)

        for config_index in config_indices:
            partial_config = BLACKWELL_DECOMPOSE_K_PARTIAL_CONFIGS[config_index]
            m_tiles = math.ceil(m / partial_config.block_m)
            if partial_config.two_ctas:
                m_tiles = math.ceil(m_tiles / 2) * 2
            output_tiles = m_tiles * math.ceil(n / partial_config.block_n)
            waves = 1 if partial_config.two_ctas else 2
            k_split = max(
                2,
                round(waves * device.multi_processor_count / output_tiles),
            )
            k_split = min(k_split, k // partial_config.block_k)
            while k_split >= 2:
                k_part = (
                    math.ceil(math.ceil(k / k_split) / partial_config.block_k)
                    * partial_config.block_k
                )
                workspace_bytes = k_split * m_tiles * partial_config.block_m * n * 4
                if (k_split - 1) * k_part < k and workspace_bytes <= 128 * 1024**2:
                    break
                k_split -= 1
            if k_split >= 2:
                yield {"k_split": k_split, "config_index": config_index}
