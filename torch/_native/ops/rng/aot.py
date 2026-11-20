from __future__ import annotations

from pathlib import Path
from typing import Any

from torchgen.native_aot_decl import load_by_path, NativeCoverage


_CFG = load_by_path(
    "rng_aot_config", str(Path(__file__).parents[2] / "cutedsl/config.py")
)


class Distribution(NativeCoverage):
    DISPATCH_KEY = "CUDA"
    KERNEL_MODULE = "cutedsl_kernels.py"
    STRUCTURED = False
    ARCHS = _CFG.pointwise_aot_archs()

    def __init__(self, kind: str) -> None:
        self.kind: str = kind
        self.ATEN_OP: str = kind + "_"

    def kernel_precompile_grid(self) -> list[dict[str, Any]]:
        return [dict(kind=self.kind, capture=[False, True], eligible=True)]

    def cpp_includes(self) -> str:
        return "\n".join(
            (
                "#include <ATen/cuda/CUDAGeneratorImpl.h>",
                "#include <ATen/ops/from_blob.h>",
                "#include <c10/cuda/CUDAGraphsC10Utils.h>",
                "#include <c10/core/impl/COW.h>",
                "#include <mutex>",
            )
        )

    def _checks(self) -> str:
        common = """
if (!self.is_cuda() || self.scalar_type() != at::kFloat || !self.is_contiguous() ||
    self.is_neg() || self.is_conj() || self._is_zerotensor() || generator.has_value() ||
    self.numel() == 0 || self.numel() > (INT32_MAX - 1) / 4 + 1 ||
    reinterpret_cast<uintptr_t>(self.const_data_ptr()) % 4 ||
    c10::impl::cow::is_cow_data_ptr(self.storage().data_ptr())) return false;
"""
        if self.kind == "uniform":
            return (
                common
                + """
const double limit = std::numeric_limits<float>::max();
if (!(from >= -limit && from <= to && to <= limit && to - from <= limit)) return false;
"""
            )
        return common + "if (!(std >= 0.0)) return false;\n"

    def cpp_covers(self) -> str:
        return self._checks() + "return true;"

    def cpp_dispatch_prelude(self) -> str:
        return (
            self._checks()
            + "const bool capture = c10::cuda::currentStreamCaptureStatusMayInitCtx() == c10::cuda::CaptureStatus::Active;"
        )

    def cpp_dispatch(self, spec: dict[str, Any]) -> str:
        return "capture" if spec["capture"] else "!capture"

    def cpp_launch(self, spec: dict[str, Any], launch_fn: str) -> str:
        if self.kind == "uniform":
            params = "const float b = float(from), c = float(to), a = c - b;"
        else:
            params = "const float a = float(std), b = float(mean), c = 0;"
        return f"""
const int64_t n = self.numel();
const int32_t grid = std::min<int64_t>((n + {_CFG.RNG_BLOCK - 1}) / {_CFG.RNG_BLOCK},
    _naot_props->multiProcessorCount * (_naot_props->maxThreadsPerMultiProcessor / {_CFG.RNG_BLOCK}));
const int32_t iterations = (n - 1) / (int64_t(grid) * {_CFG.RNG_BLOCK * 4}) + 1;
auto* gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(generator,
    at::cuda::detail::getDefaultCUDAGenerator(self.get_device()));
at::PhiloxCudaState state;
{{
  std::lock_guard<std::mutex> lock(gen->mutex_);
  state = gen->philox_cuda_state(uint64_t(iterations) * 4);
}}
TORCH_INTERNAL_ASSERT(state.captured_ == capture);
// Capture views borrow graph state; eager views are unused.
auto options = at::TensorOptions().dtype(at::kLong).device(capture ? self.device() : at::Device(at::kCPU));
auto seed = at::from_blob(capture ? static_cast<void*>(state.seed_.ptr) : &state.seed_.val, {{1}}, options);
auto offset = at::from_blob(capture ? static_cast<void*>(state.offset_.ptr) : &state.offset_.val, {{1}}, options);
{params}
{launch_fn}(self.view({{-1}}), seed, offset,
    capture ? 0 : int64_t(state.seed_.val), capture ? 0 : int64_t(state.offset_.val),
    int64_t(state.offset_intragraph_), a, b, c, iterations, grid, at::cuda::getCurrentCUDAStream());
aot_result = self;
"""


def declarations() -> list[Distribution]:
    return [Distribution("uniform"), Distribution("normal")]
