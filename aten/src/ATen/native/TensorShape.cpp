#include <vector>

#include "ATen/ATen.h"
#include "ATen/ExpandUtils.h"
#include "ATen/NativeFunctions.h"
#include "ATen/WrapDimUtils.h"

namespace at {
namespace native {

std::vector<Tensor> chunk(const Tensor& self, int64_t chunks, int64_t dim) {
  int64_t split_size = (self.size(dim) + chunks - 1) / chunks;
  // ensure this is dispatched through Tensor/Type, rather than the native function directly.
  return self.split(split_size, dim);
}

Tensor expand(const Tensor& self, IntList size) {
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

Tensor expand_as(const Tensor& self, const Tensor& other) {
  return self.expand(other.sizes());
}

Tensor narrow(const Tensor& self, int64_t dim, int64_t start, int64_t length) {
  AT_ASSERT(self.dim() > 0, "narrow() cannot be applied to a 0-dim tensor.");
  auto cur_size = self.size(dim);
  if (start < 0 || start >= cur_size) {
    runtime_error("start out of range");
  }
  if (length <= 0 || start > cur_size - length) {
    runtime_error("length out of range");
  }
  return at::native::slice(self, dim, start, start + length, 1);
}

Tensor permute(const Tensor& self, IntList dims) {
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

Tensor select(const Tensor& self, int64_t dim, int64_t index) {
  int64_t ndim = self.dim();
  AT_ASSERT(ndim > 0, "select() cannot be applied to a 0-dim tensor.");
  dim = maybe_wrap_dim(dim, ndim);
  auto size = self.size(dim);
  if (index < -size || index >= size) {
    std::stringstream ss;
    ss << "select(): index " << index << " out of range for tensor of size ";
    ss << self.sizes() << " at dimension " << dim;
    throw std::runtime_error(ss.str());
  }
  if (index < 0) {
    index += size;
  }
  auto sizes = std::vector<int64_t>(self.sizes());
  auto strides = std::vector<int64_t>(self.strides());
  auto storage_offset = self.storage_offset() + index * strides[dim];
  sizes.erase(sizes.begin() + dim);
  strides.erase(strides.begin() + dim);
  return self.as_strided(sizes, strides, storage_offset);
}

Tensor slice(const Tensor& self, int64_t dim, int64_t start, int64_t end, int64_t step) {
  int64_t ndim = self.dim();
  AT_ASSERT(ndim > 0, "slice() cannot be applied to a 0-dim tensor.");
  dim = maybe_wrap_dim(dim, ndim);
  auto sizes = std::vector<int64_t>(self.sizes());
  auto strides = std::vector<int64_t>(self.strides());
  if (step <= 0) {
    // TODO: support negative strides
    throw std::runtime_error("slice step must be positive");
  }
  if (start < 0) {
    start += sizes[dim];
  }
  if (end < 0) {
    end += sizes[dim];
  }
  if (start < 0) {
    start = 0;
  } else if (start >= sizes[dim]) {
    start = sizes[dim];
  }
  if (end < start) {
    end = start;
  } else if (end >= sizes[dim]) {
    end = sizes[dim];
  }
  auto storage_offset = self.storage_offset() + start * strides[dim];
  auto len = end - start;
  sizes[dim] = (len + step - 1) / step;  // round-up
  strides[dim] *= step;
  return self.as_strided(sizes, strides, storage_offset);
}

std::vector<Tensor> split(const Tensor& self, int64_t split_size, int64_t dim) {
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

Tensor stack(TensorList tensors, int64_t dim) {
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

std::tuple<std::vector<int64_t>, std::vector<int64_t> >
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

std::tuple<std::vector<int64_t>, std::vector<int64_t> >
inferSqueezeGeometry(const Tensor& tensor, const IntList& dim_list) {
  if (tensor.dim() == 0) {
    throw std::runtime_error("cannot squeeze tensor without dimensions");
  }

  std::vector<bool> squeeze_mask(tensor.dim());
  for (size_t i = 0; i < dim_list.size(); ++i) {
    const int64_t d = maybe_wrap_dim(dim_list[i], tensor.dim());
    if (tensor.sizes()[d] == 1) squeeze_mask[d] = true;
  }

  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  for(int64_t d = 0; d < tensor.dim(); d++) {
    if (!squeeze_mask[d]) {
      sizes.push_back(tensor.sizes()[d]);
      strides.push_back(tensor.strides()[d]);
    }
  }
  return std::make_tuple(sizes, strides);
}

std::tuple<std::vector<int64_t>, std::vector<int64_t> >
inferUnsqueezeGeometry(const Tensor& tensor, const IntList& dims) {
  if (tensor.numel() == 0) {
    throw std::runtime_error("cannot unsqueeze empty tensor");
  }

  // Note the dims could refer to positions outside the dimensionality of the
  // tensor. E.g., for a tensor of shape {2, 2} dims={2, 3, 4} is valid.
  // In general, the behavior of unsqueeze is defined by behavior of squeeze
  // in a sense, that t.unsqeeze(dims).squeeze(dims) must end up in t for all
  // possible dims.
  std::vector<bool> unsqueeze_mask(dims.size() + tensor.dim());
  for (size_t i = 0; i < dims.size(); ++i) {
    const uint64_t dim = maybe_wrap_dim(dims[i], unsqueeze_mask.size());
    if (unsqueeze_mask[dim]) {
      throw std::runtime_error("unsqueeze dims must be unique");
    }
    unsqueeze_mask[dim] = true;
  }

  std::vector<int64_t> sizes, strides;
  sizes.reserve(unsqueeze_mask.size());
  strides.reserve(unsqueeze_mask.size());
  for (size_t i = 0, read_index = 0; i < unsqueeze_mask.size(); ++i) {
    if (unsqueeze_mask[i]) {
      sizes.push_back(1);
      strides.push_back(read_index >= static_cast<size_t>(tensor.dim())
                            ? 1
                            : tensor.sizes()[read_index] *
                                  tensor.strides()[read_index]);
    } else {
      sizes.push_back(tensor.sizes()[read_index]);
      strides.push_back(tensor.strides()[read_index]);
      ++read_index;
    }
  }
  return std::make_tuple(sizes, strides);
}

Tensor squeeze(const Tensor& self) {
  auto g = inferSqueezeGeometry(self);
  return self.as_strided(std::get<0>(g), std::get<1>(g));
}

Tensor squeeze(const Tensor& self, IntList dims) {
  if (self.dim() == 0) return self;
  auto g = inferSqueezeGeometry(self, dims);
  return self.as_strided(std::get<0>(g), std::get<1>(g));
}

Tensor squeeze(const Tensor& self, int64_t dim) {
  return at::native::squeeze(self, IntList(&dim, 1));
}

Tensor & squeeze_(Tensor& self, int64_t dim) {
  return at::native::squeeze_(self, IntList(&dim, 1));
}

Tensor & squeeze_(Tensor& self) {
  auto g = inferSqueezeGeometry(self);
  return self.as_strided_(std::get<0>(g), std::get<1>(g));
}

Tensor & squeeze_(Tensor& self, IntList dims) {
  if (self.dim() == 0) return self;
  auto g = inferSqueezeGeometry(self, dims);
  return self.as_strided_(std::get<0>(g), std::get<1>(g));
}

Tensor unsqueeze(const Tensor& self, int64_t dim) {
  return at::native::unsqueeze(self, IntList(&dim, 1));
}

Tensor & unsqueeze_(Tensor& self, int64_t dim) {
  return at::native::unsqueeze_(self, IntList(&dim, 1));
}

Tensor unsqueeze(const Tensor& self, IntList dims) {
  auto g = inferUnsqueezeGeometry(self, dims);
  return self.as_strided(std::get<0>(g), std::get<1>(g));
}

Tensor & unsqueeze_(Tensor& self, IntList dims) {
  auto g = inferUnsqueezeGeometry(self, dims);
  return self.as_strided_(std::get<0>(g), std::get<1>(g));
}

Tensor view_as(const Tensor& self, const Tensor& other) {
  return self.view(other.sizes());
}

}  // namespace native
}  // namespace at
