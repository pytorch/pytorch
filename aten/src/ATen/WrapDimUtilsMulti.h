#pragma once

#include <c10/core/TensorImpl.h>
#include <ATen/WrapDimUtils.h>
#include <sstream>
#include <bitset>

namespace at {

// This is in an extra file to work around strange interaction of
// bitset on Windows with operator overloading

constexpr size_t dim_bitset_size = 64;

static inline std::bitset<dim_bitset_size> dim_list_to_bitset(IntArrayRef dims, int64_t ndims) {
  TORCH_CHECK(ndims <= (int64_t) dim_bitset_size, "only tensors with up to ", dim_bitset_size, " dims are supported");
  std::bitset<dim_bitset_size> seen;
  for (size_t i = 0; i < dims.size(); i++) {
    size_t dim = maybe_wrap_dim(dims[i], ndims);
    TORCH_CHECK(!seen[dim], "dim ", dim, " appears multiple times in the list of dims");
    seen[dim] = true;
  }
  return seen;
}

}
