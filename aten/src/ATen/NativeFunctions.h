#pragma once

#include <ATen/ATen.h>
#include <vector>

namespace at {
namespace native {

/*
[NativeFunction]
name: split
arg: Tensor self
arg: int64_t split_size
arg: int64_t dim
return: TensorList
variants: method, function
type_method_definition_level: base
type_method_definition_dispatch: at::native::split
[/NativeFunction]
*/
static inline std::vector<Tensor> split(const Tensor &self, int64_t split_size, int64_t dim) {
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
arg: int64_t dim
return: TensorList
variants: method, function
type_method_definition_level: base
type_method_definition_dispatch: at::native::chunk
[/NativeFunction]
*/
static inline std::vector<Tensor> chunk(const Tensor &self, int64_t chunks, int64_t dim) {
  int64_t split_size = (self.size(dim) + chunks - 1) / chunks;
  return at::native::split(self, split_size, dim);
}

}
}
