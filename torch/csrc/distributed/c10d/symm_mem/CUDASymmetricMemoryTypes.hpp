#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include <c10/util/hash.h>

#if defined(USE_ROCM)
#include <hip/hip_runtime_api.h>
#endif

namespace c10d::symmetric_memory {

// Key type for the symmetric memory map. `void*` for tensor storage ptr,
// `std::string` for group name.
using SymmMemKey = std::pair<void*, std::string>;
// Hash function for the symmetric memory map. c10::hash has a std::pair
// specialization (line 323-329 of hash.h) that delegates to the tuple hasher
// which combines hashes of each element.
using SymmMemKeyHash = c10::hash<SymmMemKey>;

// Covers NVL72
constexpr int max_cuda_p2p_domain_size = 72;
// Maximum number of channels
constexpr int symm_max_nblocks = 32;

// Maximally, a rank will need to sync with all other ranks, over all
// channels. Each signal is 32 bits, which is the minimum unit for atomic cas.
// Default signal pad size, can be overridden via set_signal_pad_size().
constexpr size_t default_signal_pad_size =
    symm_max_nblocks * max_cuda_p2p_domain_size * sizeof(uint32_t);

#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
using HandleType = CUmemGenericAllocationHandle;
#elif defined(USE_ROCM)
using HandleType = hipMemGenericAllocationHandle_t;
#else
using HandleType = void*;
#endif

} // namespace c10d::symmetric_memory
