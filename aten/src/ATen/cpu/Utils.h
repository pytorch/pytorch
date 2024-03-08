#pragma once

#include <c10/macros/Export.h>

namespace at::cpu {

// Detect if CPU support Vector Neural Network Instruction.
TORCH_API bool does_cpu_support_vnni();

// Detect if CPU support AVX512BF16 ISA.
TORCH_API bool does_cpu_support_avx512bf16();

} // namespace at::cpu
