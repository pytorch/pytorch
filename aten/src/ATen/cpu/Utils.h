#pragma once

#include <cstdint>

#include <c10/macros/Export.h>

namespace at::cpu {

TORCH_API bool is_avx2_supported();
TORCH_API bool is_avx512_supported();

// Detect if CPU support Vector Neural Network Instruction.
TORCH_API bool is_avx512_vnni_supported();

// Detect if CPU supports AVX512_BF16 ISA
TORCH_API bool is_avx512_bf16_supported();

// Detect if CPU support Advanced Matrix Extension.
TORCH_API bool is_amx_tile_supported();

// Enable the system to use AMX instructions.
TORCH_API bool init_amx();

// Detect if CPU supports Arm(R) architecture SVE ISA
TORCH_API bool is_arm_sve_supported();

// Get the max SVE vector length on Arm(R) architecture SVE ISA supported CPU's
TORCH_API uint32_t get_max_arm_sve_length();

// Get the L1 cache size per core in Byte
TORCH_API uint32_t L1d_cache_size();

// Get the L2 cache size per core in Byte
TORCH_API uint32_t L2_cache_size();

} // namespace at::cpu
