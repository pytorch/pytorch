#pragma once

#include <c10/metal/common.h>

C10_METAL_CONSTEXPR uint32_t kLstmGates = 4;

// Threads don't interact, so this only caps the threadgroup. Measured up to 7%
// faster than the default of 1024.
C10_METAL_CONSTEXPR uint32_t kCellMaxThreadsPerThreadgroup = 128;

// A batched at::mm_out runs MPSGraph on every timestep and closes the cell's
// encoder, so batches use the Metal kernels. A single row keeps ATen's native
// GEMV, and reductions above this size measured slower on the Metal kernels.
C10_METAL_CONSTEXPR uint32_t kMetalMmMaxReduction = 512;
