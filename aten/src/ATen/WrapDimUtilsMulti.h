#pragma once

#include <ATen/WrapDimUtils.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/irange.h>
#include <bitset>
#include <sstream>

namespace at {

// This is in an extra file to work around strange interaction of
// bitset on Windows with operator overloading

constexpr size_t dim_bitset_size = 64;

static inline std::bitset<dim_bitset_size> dim_list_to_bitset(
    OptionalIntArrayRef opt_dims,
    int64_t ndims) {
  TORCH_CHECK(
      ndims <= (int64_t)dim_bitset_size,
      "only tensors with up to ",
      dim_bitset_size,
      " dims are supported");
  std::bitset<dim_bitset_size> seen;
  if (opt_dims.has_value()) {
    auto dims = opt_dims.value();
    for (const auto i : c10::irange(dims.size())) {
      size_t dim = maybe_wrap_dim(dims[i], ndims);
      TORCH_CHECK(
          !seen[dim],
          "dim ",
          dim,
          " appears multiple times in the list of dims");
      seen[dim] = true;
    }
  } else {
    for (int64_t dim = 0; dim < ndims; dim++) {
      seen[dim] = true;
    }
  }
  return seen;
}

} // namespace at
