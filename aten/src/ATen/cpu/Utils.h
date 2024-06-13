#pragma once

#include <c10/macros/Export.h>

namespace at::cpu {

TORCH_API bool is_cpu_support_avx2();
TORCH_API bool is_cpu_support_avx512();

// Detect if CPU support Vector Neural Network Instruction.
TORCH_API bool is_cpu_support_avx512_vnni();

} // namespace at::cpu
