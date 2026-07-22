#pragma once

#include <c10/macros/Export.h>
#include <cstdint>
#include <initializer_list>

namespace at::native::scaled {

#ifdef USE_ROCM
// True if the current device's GPU arch is one of the archs supporting
// torch._scaled_mm / torch._scaled_grouped_mm (MI300+).
TORCH_CUDA_CPP_API
bool rocm_scaled_mm_arch_allowed();
#else
// SM generations relevant to torch._scaled_mm / torch._scaled_grouped_mm
// device gating. Values match the corresponding SM major version.
enum class CudaScaledMmArch : int64_t {
  Sm90 = 9,
  Sm100 = 10,
};

// True if the current device's SM major version matches one of
// `required_archs`. With the default (empty) argument, this is the general
// torch._scaled_mm gate: SM >= 9.0, or exactly SM 8.9 (Ada/L4).
TORCH_CUDA_CPP_API
bool cuda_scaled_mm_arch_allowed(std::initializer_list<CudaScaledMmArch> required_archs = {});
#endif

} // namespace at::native::scaled
