# mypy: allow-untyped-defs

"""
Fused gate+up GEMM + SiLU kernel for Llama MLP.

Replaces: silu(x @ gate_weight.T) * (x @ up_weight.T)
With:     xpu._fused_gate_up_silu(x, gate_weight, up_weight)

The XPU kernel fuses the two GEMMs into one (via weight concatenation)
and applies SiLU*mul in a single SYCL kernel, eliminating intermediate
memory traffic.

Fallback: decomposed silu(mm) * mm when not XPU or not fp16/bf16.
"""

import logging

import torch

from ..lowering import lowerings
from ..select_algorithm import autotune_select_algorithm, ExternKernelChoice


log = logging.getLogger(__name__)

aten = torch.ops.aten


def _is_fused_gate_up_silu_available():
    if not hasattr(torch.ops, "xpu") or not hasattr(
        torch.ops.xpu, "_is_fused_gate_up_silu_available"
    ):
        return False
    return torch.ops.xpu._is_fused_gate_up_silu_available()


def tuned_fused_gate_up_silu(input, gate_weight, up_weight, *, layout=None):
    """
    Computes silu(input @ gate_weight.T) * (input @ up_weight.T).

    Uses the fused XPU kernel when available, falls back to decomposed ops.
    """
    from ..ir import FixedLayout
    from .mm_common import realize_inputs

    input, gate_weight, up_weight = [
        realize_inputs(t) for t in (input, gate_weight, up_weight)
    ]

    M = input.get_size()[0]
    N = gate_weight.get_size()[0]
    device = input.get_device()
    dtype = input.get_dtype()

    # Fallback: decompose into individual ops
    if (
        not _is_fused_gate_up_silu_available()
        or str(device.type) != "xpu"
        or dtype not in (torch.float16, torch.bfloat16)
    ):
        # Decomposed fallback: t + mm + silu + t + mm + mul
        gate_t = lowerings[aten.t](gate_weight)
        up_t = lowerings[aten.t](up_weight)
        gate_out = lowerings[aten.mm](input, gate_t)
        up_out = lowerings[aten.mm](input, up_t)
        gate_silu = lowerings[aten.silu](gate_out)
        return lowerings[aten.mul](gate_silu, up_out)

    # Wrap the torch-xpu-ops kernel as an autotuning candidate (deferred to
    # avoid AttributeError on non-XPU builds where the op is not registered).
    # Use lookup() to avoid duplicate registration on repeated compilations.
    xpu_fused_gate_up_silu = ExternKernelChoice.lookup(
        "_fused_gate_up_silu"
    ) or ExternKernelChoice(
        torch.ops.xpu._fused_gate_up_silu,
        "xpu::_fused_gate_up_silu",
        has_out_variant=False,
    )

    layout = FixedLayout(device, dtype, [M, N])

    choices = [
        xpu_fused_gate_up_silu.bind((input, gate_weight, up_weight), layout),
    ]

    # No Triton alternative for now — the SYCL kernel is the only choice.
    # Future: add a Triton template for XPU when Triton-on-XPU supports
    # fused GEMM+activation.

    node, _ = autotune_select_algorithm(
        "fused_gate_up_silu",
        choices,
        [input, gate_weight, up_weight],
        layout,
    )
    return node
