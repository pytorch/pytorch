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
name: size
arg: Tensor self
arg: int64_t dim
return: int64_t
variants: method, function
type_method_definition_level: base
type_method_definition_dispatch: at::native::size
[/NativeFunction]
*/
static inline int64_t size(const Tensor &self, int64_t dim) {
  dim = maybe_wrap_dim(dim, self.dim());
  // wrap_dim guarantees bounds are correct.
  return self.sizes()[dim];
}

/*
[NativeFunction]
name: stride
arg: Tensor self
arg: int64_t dim
return: int64_t
variants: method, function
type_method_definition_level: base
type_method_definition_dispatch: at::native::stride
[/NativeFunction]
*/
static inline int64_t stride(const Tensor &self, int64_t dim) {
  dim = maybe_wrap_dim(dim, self.dim());
  // wrap_dim guarantees bounds are correct.
  return self.strides()[dim];
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
arg: IntList size
return: Tensor
variants: method, function
type_method_definition_level: base
type_method_definition_dispatch: at::native::expand
[/NativeFunction]
*/
static inline Tensor expand(const Tensor &self, IntList size) {
  if (size.size() < (size_t)self.dim()) {
    std::ostringstream ss;
    ss << "expand(" << self.type() << "{" << self.sizes() << "}, size=" << size
       << "): the number of sizes provided (" << size.size() << ") "
       << "must be greater or equal to the number of dimensions in the tensor ("
       << self.dim() << ")";
    throw std::runtime_error(ss.str());
  }

  std::vector<int64_t> expandedSizes;
  std::vector<int64_t> expandedStrides;
  std::tie(expandedSizes, expandedStrides) = inferExpandGeometry(self, size);

  return self.as_strided(expandedSizes, expandedStrides);
}

static inline std::tuple<std::vector<int64_t>, std::vector<int64_t> >
inferSqueezeGeometry(const Tensor &tensor) {
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;

  for(int64_t d = 0; d < tensor.dim(); d++) {
    if(tensor.sizes()[d] != 1) {
      sizes.push_back(tensor.sizes()[d]);
      strides.push_back(tensor.strides()[d]);
    }
  }

  return std::make_tuple(sizes, strides);
}

static inline std::tuple<std::vector<int64_t>, std::vector<int64_t> >
inferSqueezeGeometry(const Tensor &tensor, int64_t dim) {
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;

  for(int64_t d = 0; d < tensor.dim(); d++) {
    if(d != dim || tensor.sizes()[dim] != 1) {
      sizes.push_back(tensor.sizes()[d]);
      strides.push_back(tensor.strides()[d]);
    }
  }
  return std::make_tuple(sizes, strides);
}

static inline std::tuple<std::vector<int64_t>, std::vector<int64_t> >
inferUnsqueezeGeometry(const Tensor &tensor, int64_t dim) {
  if (tensor.numel() == 0) {
    throw std::runtime_error("cannot unsqueeze empty tensor");
  }

  std::vector<int64_t> sizes(tensor.sizes());
  std::vector<int64_t> strides(tensor.strides());
  int64_t new_stride = dim >= tensor.dim() - 1 ? 1 : sizes[dim] * strides[dim];
  sizes.insert(sizes.begin() + dim, 1);
  strides.insert(strides.begin() + dim, new_stride);

  return std::make_tuple(sizes, strides);
}

/*
[NativeFunction]
name: squeeze
arg: Tensor self
return: Tensor
variants: method, function
type_method_definition_level: base
type_method_definition_dispatch: at::native::squeeze
[/NativeFunction]
*/
static inline Tensor squeeze(const Tensor & self) {
  auto g = inferSqueezeGeometry(self);
  return self.as_strided(std::get<0>(g), std::get<1>(g));
}

/*
[NativeFunction]
name: squeeze
arg: Tensor self
arg: int64_t dim
return: Tensor
variants: method, function
type_method_definition_level: base
type_method_definition_dispatch: at::native::squeeze
[/NativeFunction]
*/
static inline Tensor squeeze(const Tensor & self, int64_t dim) {
  dim = maybe_wrap_dim(dim, self.dim());

  if (self.sizes()[dim] != 1) {
    return self.as_strided(self.sizes().vec(), self.strides().vec());
  }
  auto g = inferSqueezeGeometry(self, dim);
  return self.as_strided(std::get<0>(g), std::get<1>(g));
}

/*
[NativeFunction]
name: squeeze_
arg: Tensor self
return: Tensor
variants: method, function
type_method_definition_level: base
type_method_definition_dispatch: at::native::squeeze_
[/NativeFunction]
*/
static inline Tensor & squeeze_(Tensor & self) {
  auto g = inferSqueezeGeometry(self);
  return self.as_strided_(std::get<0>(g), std::get<1>(g));
}

/*
[NativeFunction]
name: squeeze_
arg: Tensor self
arg: int64_t dim
return: Tensor
variants: method, function
type_method_definition_level: base
type_method_definition_dispatch: at::native::squeeze_
[/NativeFunction]
*/
static inline Tensor & squeeze_(Tensor & self, int64_t dim) {
  dim = maybe_wrap_dim(dim, self.dim());

  if (self.sizes()[dim] != 1) {
    return self.as_strided_(self.sizes().vec(), self.strides().vec());
  }
  auto g = inferSqueezeGeometry(self, dim);
  return self.as_strided_(std::get<0>(g), std::get<1>(g));
}

/*
[NativeFunction]
name: unsqueeze
arg: Tensor self
arg: int64_t dim
return: Tensor
variants: method, function
type_method_definition_level: base
type_method_definition_dispatch: at::native::unsqueeze
[/NativeFunction]
*/
static inline Tensor unsqueeze(const Tensor & self, int64_t dim) {
  dim = maybe_wrap_dim(dim, self.dim() + 1);

  auto g = inferUnsqueezeGeometry(self, dim);
  return self.as_strided(std::get<0>(g), std::get<1>(g));
}

/*
[NativeFunction]
name: unsqueeze_
arg: Tensor self
arg: int64_t dim
return: Tensor
variants: method, function
type_method_definition_level: base
type_method_definition_dispatch: at::native::unsqueeze_
[/NativeFunction]
*/
static inline Tensor & unsqueeze_(Tensor & self, int64_t dim) {
  dim = maybe_wrap_dim(dim, self.dim() + 1);

  auto g = inferUnsqueezeGeometry(self, dim);
  return self.as_strided_(std::get<0>(g), std::get<1>(g));
}

/*
[NativeFunction]
name: stack
arg: TensorList tensors
arg: int64_t dim=0
return: Tensor
variants: function
type_method_definition_level: base
type_method_definition_dispatch: at::native::stack
[/NativeFunction]
*/
static inline Tensor stack(TensorList tensors, int64_t dim=0) {
  if (tensors.size() == 0) {
    throw std::runtime_error("stack expects a non-empty TensorList");
  }
  dim = maybe_wrap_dim(dim, tensors[0].dim() + 1);

  std::vector<Tensor> inputs(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    inputs[i] = tensors[i].unsqueeze(dim);
  }
  return at::cat(inputs, dim);
}

}
}
