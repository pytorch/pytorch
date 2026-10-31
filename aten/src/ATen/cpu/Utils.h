#pragma once

#include <string>
#include <unordered_map>

#include <ATen/core/ivalue.h>
#include <c10/macros/Export.h>

// Full RVV FP64 vector support. __riscv_v_intrinsic alone is defined for the
// Zve32x/Zve32f/Zve64f sub-profiles too, which lack the vfloat64m2_t intrinsics
// that the FP64 RVV kernels use, so also require FP64 element width. This is a
// compile-time flag, so other TUs gate RVV code on the same condition here.
#if defined(__riscv_v_intrinsic) && __riscv_v_intrinsic >= 12000 && \
    defined(__riscv_v_elen_fp) && __riscv_v_elen_fp >= 64
#define TORCH_RVV_FP64_ENABLED 1
#endif

namespace at::cpu {

// Returns a map of CPU capabilities detected at runtime via cpuinfo.
// Keys are capability names (e.g., "avx2", "neon"), values are bools
// for ISA flags, integers for cache sizes/core counts, or strings
// for architecture/CPU name.
TORCH_API std::unordered_map<std::string, c10::IValue> get_cpu_capabilities();

// Whether this build was compiled with the FP64 RVV kernels enabled (see
// TORCH_RVV_FP64_ENABLED above).
TORCH_API bool has_rvv();

// Detect if CPU supports AVX512 Vector Neural Network Instructions.
TORCH_API bool is_avx512_vnni_supported();

// Enable the system to use AMX instructions.
TORCH_API bool init_amx();

} // namespace at::cpu
