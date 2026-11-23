#pragma once

#include <ATen/ExpandUtils.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/core/Tensor.h>
#include <c10/util/safe_numerics.h>
#include <c10/util/SmallVector.h>
#include <c10/util/irange.h>
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

// Validates a tensor bound and converts it to dtype. The kernels write self
// while the bound is still read, so a bound sharing self's storage is cloned:
// assert_no_overlap lets through aliases it can't analyze (e.g. a non-dense
// view of self).
inline Tensor philox_prepare_bound(
    const char* op_name, const char* name, const Tensor& bound, const Tensor& self, ScalarType dtype) {
  TORCH_CHECK(bound.is_floating_point(),
      op_name, ": ", name, " must be a floating point tensor, got ",
      bound.scalar_type());
  TORCH_CHECK(is_expandable_to(bound.sizes(), self.sizes()),
      op_name, ": ", name, " shape ", bound.sizes(),
      " is not broadcastable to the output shape ", self.sizes());
  at::assert_no_overlap(self, bound);
  auto converted = bound.to(dtype);
  return converted.is_alias_of(self) ? converted.clone() : converted;
}

// Where a shard (the output) sits within a row-major global tensor that a
// single key fills. The shard splits into runs: stretches that are contiguous
// in the global tensor. Trailing dims the shard fully covers are merged so each
// run is as long as possible.
struct PhiloxShardLayout {
  int64_t run_len = 1;
  int64_t num_runs = 1;
  // Global flat index of the shard's first element.
  uint64_t base = 0;
  // Sizes of the run grid and their strides in the global tensor, innermost
  // first. Decomposing a run index over these gives the global flat index of
  // the run's first element, relative to base.
  c10::SmallVector<int64_t, 8> grid_sizes;
  c10::SmallVector<int64_t, 8> grid_strides;
};

inline PhiloxShardLayout philox_shard_layout(
    const char* op_name, const Tensor& self, const Tensor& key,
    IntArrayRef global_shape, IntArrayRef global_offset) {
  TORCH_CHECK(key.scalar_type() == kUInt64,
      op_name, ": key must have dtype uint64, got ", key.scalar_type());
  TORCH_CHECK(self.device() == key.device(),
      op_name, ": self and key must be on the same device, got ",
      self.device(), " and ", key.device());
  TORCH_CHECK(key.dim() == 1 && key.size(0) == 2,
      op_name, ": key must have shape (2,), got shape ", key.sizes());
  at::assert_no_internal_overlap(self);
  const int64_t ndim = self.dim();
  TORCH_CHECK(
      static_cast<int64_t>(global_shape.size()) == ndim &&
          static_cast<int64_t>(global_offset.size()) == ndim,
      op_name, ": global_shape and global_offset must have one entry per output "
      "dim, got ", global_shape, " and ", global_offset, " for output shape ",
      self.sizes());
  int64_t global_numel = 1;
  for (const auto d : c10::irange(ndim)) {
    // Subtract rather than add so huge offsets can't overflow.
    TORCH_CHECK(
        global_offset[d] >= 0 && global_shape[d] >= 0 &&
            self.size(d) <= global_shape[d] - global_offset[d],
        op_name, ": output of shape ", self.sizes(), " at global_offset ",
        global_offset, " does not fit in global_shape ", global_shape);
    // Bounds every product in the layout below, which are all <= global_numel.
    TORCH_CHECK(!c10::mul_overflows(global_numel, global_shape[d], &global_numel),
        op_name, ": global_shape ", global_shape, " has too many elements");
  }

  PhiloxShardLayout layout;
  if (ndim == 0) {
    return layout;
  }
  int64_t j = ndim - 1;
  while (j > 0 && self.size(j) == global_shape[j]) {
    j--;
  }
  int64_t inner = 1;
  for (int64_t d = ndim - 1; d > j; d--) {
    inner *= global_shape[d];
  }
  layout.run_len = self.size(j) * inner;
  layout.base = static_cast<uint64_t>(global_offset[j] * inner);
  int64_t stride = global_shape[j] * inner;
  for (int64_t d = j - 1; d >= 0; d--) {
    layout.grid_sizes.push_back(self.size(d));
    layout.grid_strides.push_back(stride);
    layout.num_runs *= self.size(d);
    layout.base += static_cast<uint64_t>(global_offset[d] * stride);
    stride *= global_shape[d];
  }
  return layout;
}

} // namespace at::native
