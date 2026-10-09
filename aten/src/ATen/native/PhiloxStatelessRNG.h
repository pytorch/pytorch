#pragma once

#include <ATen/ExpandUtils.h>
#include <ATen/core/Tensor.h>
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

// Checks that a tensor bound can supply a value for each element of self.
inline void philox_check_bound(
    const char* op_name, const char* name, const Tensor& bound, const Tensor& self) {
  TORCH_CHECK(bound.is_floating_point(),
      op_name, ": ", name, " must be a floating point tensor, got ",
      bound.scalar_type());
  TORCH_CHECK(is_expandable_to(bound.sizes(), self.sizes()),
      op_name, ": ", name, " shape ", bound.sizes(),
      " is not broadcastable to the output shape ", self.sizes());
}

} // namespace at::native
