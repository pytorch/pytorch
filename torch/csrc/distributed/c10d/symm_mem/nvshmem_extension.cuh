#pragma once

#include <c10/macros/Macros.h>

#define NVSHMEM_CHECK(stmt, msg)                                             \
  do {                                                                       \
    int result = (stmt);                                                     \
    TORCH_CHECK(                                                             \
        result == 0,                                                         \
        std::string(__FILE__) + ":" + std::to_string(__LINE__) + " " + msg + \
            ". Error code: " + std::to_string(result));                      \
  } while (0)

namespace c10d::nvshmem_extension {

// Check if NVSHMEM is available
TORCH_API bool is_nvshmem_available();

// Initializes the device state in CUmodule so that it’s able to perform NVSHMEM
// operations.
TORCH_API void nvshmemx_cumodule_init(uintptr_t module);

} // namespace c10d::nvshmem_extension
