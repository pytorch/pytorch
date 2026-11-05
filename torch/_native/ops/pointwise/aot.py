"""AOT instantiations for the initial floating pointwise TMA operations.

The structured hook runs after ATen meta, so its output layout is authoritative.
Unsupported calls return false and continue through ATen's original CUDA kernel.
This module must remain torch-free at import time for stage-1 code generation.
"""

from __future__ import annotations

import itertools
import os
from typing import Any

from torchgen.native_aot_decl import load_by_path


# Stage 1 loads this declaration before torch can be imported.
_limits = load_by_path(
    "pointwise_limits", os.path.join(os.path.dirname(__file__), "_limits.py")
)

_DTYPES = {
    "float16": "at::kHalf",
    "bfloat16": "at::kBFloat16",
    "float32": "at::kFloat",
}
_OPS = (
    ("add", "_add", 2),
    ("sub", "_sub", 2),
    ("mul", "_mul", 2),
    ("div.Tensor", "_div", 2),
    ("neg", "_neg", 1),
    ("gt.Tensor", "_gt", 2),
)


def _axes(inputs: tuple[Any, ...], out: Any, boolean: bool) -> dict[str, Any]:
    import torch
    from torch._native.ops.pointwise._layout import _layout_config, _OperandLayout

    # Share pure layout selection with JIT without importing the CuTeDSL compiler.
    dtypes = (torch.float16, torch.bfloat16, torch.float32)
    if not all(isinstance(t, torch.Tensor) for t in inputs):
        return {"eligible": False}
    ref = inputs[0]
    if not ref.is_cuda or ref.ndim != 2 or all(t.is_contiguous() for t in inputs):
        return {"eligible": False}
    if torch.cuda.get_device_capability(ref.device)[0] != 10:
        return {"eligible": False}
    if any(
        t.layout != torch.strided
        or t.device != ref.device
        or t.dtype not in dtypes
        or t.dtype != ref.dtype
        or t.is_neg()
        or t.is_conj()
        for t in inputs
    ):
        return {"eligible": False}
    if out is not None and (
        out.layout != torch.strided
        or out.device != ref.device
        or out.shape != ref.shape
        or out.dtype != (torch.bool if boolean else ref.dtype)
        or out.is_neg()
        or out.is_conj()
        or any(torch._C._overlaps(out, t) for t in inputs)
    ):
        return {"eligible": False}
    layouts = tuple(
        _OperandLayout(
            tuple(t.shape),
            t.stride(),
            t.element_size(),
            t.const_data_ptr() % _limits._TMA_ALIGNMENT,
            t.is_contiguous(),
        )
        for t in inputs
    )
    address = 0 if out is None else out.const_data_ptr() % _limits._MIN_ALIGNMENT
    if out is None:
        # Before meta, use the reference orientation, like the C++ coverage probe.
        strides = (1, ref.size(0)) if ref.stride(0) == 1 else (ref.size(1), 1)
    else:
        strides = out.stride()
    config = _layout_config(
        layouts,
        tuple(ref.shape),
        address,
        torch.cuda.get_device_properties(ref.device).shared_memory_per_block_optin,
        strides,
    )
    if config is None:
        return {"eligible": False}
    return {"eligible": True, "dtype": ref.dtype, "transposed": config.transposed}


class _Declaration:
    DISPATCH_KEY = "CUDA"
    KERNEL_MODULE = "aot_kernels"
    ARCHS = ("sm_100f",)

    def __init__(self, aten: str, fn: str, nin: int) -> None:
        self.ATEN_OP = aten
        self.fn = fn
        self.nin = nin
        self.boolean = fn == "_gt"
        self.alpha = fn in ("_add", "_sub")

    def kernel_precompile_grid(self) -> list[dict[str, Any]]:
        return [
            {
                "dtype": list(_DTYPES),
                "transposed": mask,
                "fn": self.fn,
                "boolean": self.boolean,
                "eligible": True,
            }
            for mask in itertools.product((False, True), repeat=self.nin)
        ]

    def covered_axes(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        alpha = kwargs.get("alpha", args[self.nin] if len(args) > self.nin else 1)
        if self.alpha and isinstance(alpha, complex):
            return {"eligible": False}
        return _axes(args[: self.nin], kwargs.get("out"), self.boolean)

    def _inputs(self) -> str:
        return "{self}" if self.nin == 1 else "{self, other}"

    def cpp_helpers(self) -> str:
        return _CPP_HELPERS

    def cpp_dispatch_prelude(self) -> str:
        alpha_gate = "if (alpha.isComplex()) return false;" if self.alpha else ""
        return f"""
if (_naot_props->major != 10) return false;
{alpha_gate}
auto cfg = pointwise_tma_config({self._inputs()}, &out, {str(self.boolean).lower()},
                                _naot_props->sharedMemPerBlockOptin);
if (cfg.mask < 0) return false;
"""

    def cpp_dispatch(self, spec: dict[str, Any]) -> str:
        mask = sum(
            (1 if value else 0) << i for i, value in enumerate(spec["transposed"])
        )
        return f"self.scalar_type() == {_DTYPES[spec['dtype']]} && cfg.mask == {mask}"

    def cpp_launch(self, spec: dict[str, Any], launch_fn: str) -> str:
        views = []
        for i, transposed in enumerate(spec["transposed"]):
            name = "self" if i == 0 else "other"
            shape = "{cfg.rows, cfg.cols}"
            stride = (
                f"{{1, {name}.stride(cfg.fast)}}"
                if transposed
                else f"{{{name}.stride(1 - cfg.fast), 1}}"
            )
            views.append(f"auto x{i} = {name}.as_strided({shape}, {stride});")
        output = (
            "auto result = out.as_strided({1, cfg.rows, cfg.cols}, {0, cfg.cols, 1});"
        )
        alpha = "alpha.toFloat()" if self.alpha else "1.0f"
        operands = "x0" if self.nin == 1 else "x0, x1"
        return "\n".join(
            [
                *views,
                output,
                f"{launch_fn}({operands}, result, {alpha}, at::cuda::getCurrentCUDAStream());",
            ]
        )

    def cpp_covers(self) -> str:
        alpha_gate = "if (alpha.isComplex()) return false;" if self.alpha else ""
        return f"""
if (_naot_props->major != 10) return false;
{alpha_gate}
return pointwise_tma_config({self._inputs()}, out ? &*out : nullptr,
                            {str(self.boolean).lower()}, _naot_props->sharedMemPerBlockOptin).mask >= 0;
"""


def declarations() -> list[_Declaration]:
    return [_Declaration(*op) for op in _OPS]


_CPP_HELPERS = (
    f"""
constexpr int kPointwiseMinAlignment = {_limits._MIN_ALIGNMENT};
constexpr int kPointwiseTmaAlignment = {_limits._TMA_ALIGNMENT};
constexpr int kPointwiseBox = {_limits._TMA_BOX_SIDE};
constexpr int kPointwiseStages = {_limits._TILES_PER_BLOCK};
constexpr int kPointwiseSmemAlignment = {_limits._SMEM_ALIGNMENT};
constexpr int kPointwiseBarrierBytes = {_limits._BARRIER_BYTES};
constexpr int kPointwiseMaxGridYZ = {_limits._MAX_GRID_YZ};
"""
    + r"""
// Mirrors _axes and shared _layout_config/_tma2d_config eligibility rules.
// Keep reference/capability, output, tile limits and input views in sync.
// This post-meta native hook must work without Python or CuTeDSL.
struct PointwiseTmaConfig {
  int mask = -1;
  int fast = 1;
  int64_t rows = 0, cols = 0;
};

PointwiseTmaConfig pointwise_tma_config(
    std::initializer_list<at::Tensor> inputs, const at::Tensor* out,
    bool boolean, size_t smem_limit) {
  PointwiseTmaConfig cfg;
  const auto& ref = *inputs.begin();
  if (!ref.is_cuda() || ref.layout() != at::kStrided || ref.dim() != 2 ||
      ref.numel() <= 0 || ref.numel() > INT32_MAX) return cfg;
  auto dt = ref.scalar_type();
  if (dt != at::kHalf && dt != at::kBFloat16 && dt != at::kFloat) return cfg;
  if (std::all_of(inputs.begin(), inputs.end(), [](const auto& t) { return t.is_contiguous(); })) return cfg;
  cfg.fast = ref.stride(0) == 1 ? 0 : 1;
  if (out) {
    if (out->layout() != at::kStrided || out->device() != ref.device() || out->sizes() != ref.sizes() ||
        out->scalar_type() != (boolean ? at::kBool : dt) ||
        out->is_neg() || out->is_conj() || reinterpret_cast<uintptr_t>(out->const_data_ptr()) % kPointwiseMinAlignment != 0) return cfg;
    cfg.fast = out->stride(0) == 1 ? 0 : 1;
    if (out->stride(cfg.fast) != 1 || out->stride(1 - cfg.fast) != out->size(cfg.fast)) return cfg;
    for (const auto& t : inputs) if (out->is_alias_of(t)) return cfg;
  }
  cfg.rows = ref.size(1 - cfg.fast);
  cfg.cols = ref.size(cfg.fast);
  int box = kPointwiseBox;
  if (cfg.rows % box != 0 || cfg.cols % box != 0 || cfg.rows / box > kPointwiseMaxGridYZ ||
      inputs.size() * (kPointwiseStages * box * box * ref.element_size() + kPointwiseSmemAlignment) +
      kPointwiseStages * kPointwiseBarrierBytes > smem_limit) return cfg;
  int mask = 0, i = 0;
  for (const auto& t : inputs) {
    if (!t.is_cuda() || t.layout() != at::kStrided || t.device() != ref.device() ||
        t.scalar_type() != dt || t.sizes() != ref.sizes() ||
        t.is_neg() || t.is_conj()) return cfg;
    bool transposed = t.stride(cfg.fast) != 1;
    int inner = transposed ? 1 - cfg.fast : cfg.fast;
    int64_t pitch = t.stride(1 - inner);
    if (t.stride(inner) != 1 || pitch < t.size(inner) || pitch > INT32_MAX ||
        pitch * t.element_size() % kPointwiseMinAlignment != 0 ||
        reinterpret_cast<uintptr_t>(t.const_data_ptr()) % (transposed ? kPointwiseTmaAlignment : kPointwiseMinAlignment) != 0 ||
        (t.size(0) - 1) * t.stride(0) + (t.size(1) - 1) * t.stride(1) > INT32_MAX) return cfg;
    mask |= (transposed ? 1 : 0) << i++;
  }
  cfg.mask = mask;
  return cfg;
}
"""
)
