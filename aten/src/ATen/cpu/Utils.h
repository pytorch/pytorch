#pragma once

#include <cstdint>

#include <c10/macros/Export.h>

namespace at::cpu {

TORCH_API bool is_cpu_support_avx2();
TORCH_API bool is_cpu_support_avx512();

// Detect if CPU support Vector Neural Network Instruction.
TORCH_API bool is_cpu_support_avx512_vnni();

// Detect if CPU support Advanced Matrix Extension.
TORCH_API bool is_cpu_support_amx_tile();

// Detect if CPU support Advanced Matrix Extension for fp16.
TORCH_API bool is_cpu_support_amx_fp16();

// Enable the system to use AMX instructions.
TORCH_API bool init_amx();

// Get the L1 cache size per core in Byte
TORCH_API uint32_t L1d_cache_size();

// Get the L2 cache size per core in Byte
TORCH_API uint32_t L2_cache_size();

} // namespace at::cpu
