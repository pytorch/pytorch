#pragma once

#include <string>
#include <unordered_map>

#include <ATen/core/ivalue.h>
#include <c10/macros/Export.h>

namespace at::cpu {

// Returns a map of CPU capabilities detected at runtime via cpuinfo.
// Keys are capability names (e.g., "avx2", "neon"), values are bools
// for ISA flags, integers for cache sizes/core counts, or strings
// for architecture/CPU name.
TORCH_API std::unordered_map<std::string, c10::IValue> get_cpu_capabilities();

// Detect if CPU supports AVX512 Vector Neural Network Instructions.
TORCH_API bool is_avx512_vnni_supported();

// Enable the system to use AMX instructions.
TORCH_API bool init_amx();

} // namespace at::cpu
