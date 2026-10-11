#pragma once

#include <c10/core/Device.h>
#include <c10/macros/Export.h>

namespace at::native::scaled {

// Device architecture gate shared by the scaled matrix multiplication kernels,
// including _scaled_mm_v2 (the backend for torch.nn.functional.scaled_mm).
// On CUDA: with no flags, allows SM >= 9.0 or SM 8.9 (Ada/L4). With sm90_only
// and/or sm100_only set, checks for a matching SM major version (OR semantics
// if both are set). On ROCm, sm90_only and sm100_only are ignored.
// device_index -1 checks the current device. Only device properties are read,
// so querying another device does not switch to it or create a context there.
TORCH_CUDA_CPP_API
bool scaled_mm_arch_allowed(
    bool sm90_only = false,
    bool sm100_only = false,
    c10::DeviceIndex device_index = -1);

} // namespace at::native::scaled
