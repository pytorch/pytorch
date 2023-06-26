#pragma once

#include <c10/macros/Export.h>

namespace at {
namespace cpu {

// Detect if CPU support Vector Neural Network Instruction.
TORCH_API bool is_cpu_support_vnni();

} // namespace cpu
} // namespace at
