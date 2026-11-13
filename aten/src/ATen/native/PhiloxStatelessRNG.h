#pragma once

#include <cstdint>

// Shared constants for the stateless Philox RNG kernels. Keep these in one
// place so the CPU and CUDA kernels can't drift apart: both must agree for
// results to stay bitwise identical across devices.
namespace at::native {

// Elements produced per Philox 4x32 call: a call yields 128 bits, so 4 elements
// for 4-byte types (float/half/bfloat16/uint32) and 2 for 8-byte types
// (double/uint64). Note that we use a full float for each generated
// half/bfloat16 for better numerics.
template <typename scalar_t>
constexpr int elems_per_call = sizeof(scalar_t) == 8 ? 2 : 4;

// Largest randint range allowed for a 4-byte output. Bias from the modulo
// reduction scales as range / 2^32; this bounds it to ~6%.
constexpr uint64_t kMaxRange32 = uint64_t{1} << 28;

} // namespace at::native
