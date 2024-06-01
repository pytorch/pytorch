#pragma once

#include <c10/macros/Export.h>

namespace at::cpu {

// Detect if CPU support Vector Neural Network Instruction.
TORCH_API bool is_cpu_support_vnni();

// Enable the system to use AMX instructions.
TORCH_API bool init_amx();

} // namespace at::cpu
