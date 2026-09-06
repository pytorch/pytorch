#pragma once

#include <c10/macros/Export.h>

namespace at::native::scaled {

// True if the current device supports torch._scaled_mm / torch._scaled_grouped_mm.
// On CUDA: with no flags, allows SM >= 9.0 or SM 8.9 (Ada/L4). With sm90_only
// and/or sm100_only set, checks for a matching SM major version (OR semantics
// if both are set). On ROCm, sm90_only and sm100_only are ignored.
TORCH_CUDA_CPP_API
bool scaled_mm_arch_allowed(bool sm90_only = false, bool sm100_only = false);

} // namespace at::native::scaled
