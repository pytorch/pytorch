#pragma once

#include "ATen/ATen.h"
#include "ATen/WrapDimUtils.h"
#include "ATen/ExpandUtils.h"
#include <vector>

namespace at {
namespace native {

/*
[NativeFunction]
name: split
arg: Tensor self
arg: int64_t split_size
arg: int64_t dim=0
return: TensorList
variants: method, function
type_method_definition_level: base
type_method_definition_dispatch: at::native::split
[/NativeFunction]
*/
static inline std::vector<Tensor> split(const Tensor &self, int64_t split_size, int64_t dim=0) {
  int64_t dim_size = self.size(dim);
  int64_t num_splits = (dim_size + split_size - 1) / split_size;
  std::vector<Tensor> splits(num_splits);
  int64_t last_split_size = split_size - (split_size * num_splits - dim_size);

  for (int64_t i = 0; i < num_splits; ++i) {
    auto length = i < num_splits - 1 ? split_size : last_split_size;
    splits[i] = self.narrow(dim, i * split_size, length);
  }
  return splits;
}

/*
[NativeFunction]
name: chunk
arg: Tensor self
arg: int64_t chunks
arg: int64_t dim=0
return: TensorList
variants: method, function
type_method_definition_level: base
type_method_definition_dispatch: at::native::chunk
[/NativeFunction]
*/
static inline std::vector<Tensor> chunk(const Tensor &self, int64_t chunks, int64_t dim=0) {
  int64_t split_size = (self.size(dim) + chunks - 1) / chunks;
  // ensure this is dispatched through Tensor/Type, rather than the native function directly.
  return self.split(split_size, dim);
}

/*
[NativeFunction]
name: is_same_size
arg: Tensor self
arg: Tensor other
return: bool
variants: method, function
type_method_definition_level: base
type_method_definition_dispatch: at::native::is_same_size
[/NativeFunction]
*/
static inline bool is_same_size(const Tensor &self, const Tensor &other) {
  return self.sizes().equals(other.sizes());
}

/*
[NativeFunction]
name: permute
arg: Tensor self
arg: IntList dims
return: Tensor
variants: method, function
type_method_definition_level: base
type_method_definition_dispatch: at::native::permute
[/NativeFunction]
*/
static inline Tensor permute(const Tensor & self, IntList dims) {
  auto nDims = self.dim();
  if (dims.size() != (size_t)nDims) {
    runtime_error("number of dims don't match in permute");
  }
  auto oldSizes = self.sizes();
  auto oldStrides = self.strides();
  std::vector<int64_t> newSizes(nDims);
  std::vector<int64_t> newStrides(nDims);
  std::vector<bool> seen(nDims);
  for (int64_t i = 0; i < nDims; i++) {
    auto dim = maybe_wrap_dim(dims[i], nDims);
    if (seen[dim]) {
      runtime_error("repeated dim in permute");
    }
    seen[dim] = true;
    newSizes[i] = oldSizes[dim];
    newStrides[i] = oldStrides[dim];
  }
  return self.as_strided(newSizes, newStrides);
}

/*
[NativeFunction]
name: expand
arg: Tensor self
arg: IntList sizes
return: Tensor
variants: method, function
type_method_definition_level: base
type_method_definition_dispatch: at::native::expand
[/NativeFunction]
*/
static inline Tensor expand(const Tensor &self, IntList sizes) {
  if (sizes.size() < (size_t)self.dim()) {
    throw std::runtime_error("the number of sizes provided must be greater or equal to the "
                             "number of dimensions in the tensor");
  }

  std::vector<int64_t> expandedSizes;
  std::vector<int64_t> expandedStrides;
  std::tie(expandedSizes, expandedStrides) = inferExpandGeometry(self, sizes);

  return self.as_strided(expandedSizes, expandedStrides);
}

}
}
