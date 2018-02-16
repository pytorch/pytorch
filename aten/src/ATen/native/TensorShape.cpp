#include "ATen/ATen.h"
#include "ATen/ExpandUtils.h"
#include "ATen/NativeFunctions.h"
#include "ATen/WrapDimUtils.h"

#include <algorithm>

namespace at {
namespace native {

static void check_cat_no_zero_dim(TensorList tensors) {
  for(size_t i = 0; i < tensors.size(); ++i) {
    auto& t = tensors[i];
    if (t.dim() == 0) {
      runtime_error("zero-dimensional tensor (at position %zu) cannot be concatenated", i);
    }
  }
}

Tensor & cat_out(Tensor & result, TensorList tensors, int64_t dim) {
  check_cat_no_zero_dim(tensors);
  return at::_cat_out(result, tensors, dim);
}

Tensor cat(TensorList tensors, int64_t dim) {
  check_cat_no_zero_dim(tensors);
  return at::_cat(tensors, dim);
}

std::vector<Tensor> chunk(const Tensor& self, int64_t chunks, int64_t dim) {
  if (self.dim() == 0) {
    throw std::runtime_error("chunk expects at least a 1-dimensional tensor");
  }
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

Tensor repeat(const Tensor& self, IntList repeats) {
  if (repeats.size() < (size_t)self.dim()) {
    runtime_error("Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor");
  }

  // Add new leading dimensions to the tensor if the
  // number of target dimensions is larger than the
  // number of source dimensions.
  int64_t num_new_dimensions = repeats.size() - self.dim();
  std::vector<int64_t> padded_size(num_new_dimensions, 1);
  padded_size.insert(padded_size.end(), self.sizes().begin(), self.sizes().end());
  std::vector<int64_t> target_size(repeats.size());
  for(size_t idx = 0; idx < repeats.size(); ++idx) {
    target_size[idx] = padded_size[idx] * repeats[idx];
  }

  Tensor xtensor = self.expand(padded_size);

  Tensor result = self.type().tensor(target_size);
  Tensor urtensor = result.type().alias(result);
  for (int64_t i = 0; i < xtensor.dim(); ++i) {
    urtensor = urtensor.unfold(i, xtensor.size(i), xtensor.size(i));
  }

  urtensor.copy_(xtensor.expand_as(urtensor));

  return result;
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
  if (self.dim() == 0) {
    throw std::runtime_error("split expects at least a 1-dimensional tensor");
  }
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

static inline std::vector<Tensor> get_stack_inputs(TensorList tensors, int64_t dim) {
  std::vector<Tensor> inputs(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    inputs[i] = tensors[i].unsqueeze(dim);
  }
  return inputs;
}

Tensor stack(TensorList tensors, int64_t dim) {
  if (tensors.size() == 0) {
    throw std::runtime_error("stack expects a non-empty TensorList");
  }
  dim = maybe_wrap_dim(dim, tensors[0].dim() + 1);
  return at::cat(get_stack_inputs(tensors, dim), dim);
}

Tensor& stack_out(Tensor& result, TensorList tensors, int64_t dim) {
  if (tensors.size() == 0) {
    throw std::runtime_error("stack expects a non-empty TensorList");
  }
  dim = maybe_wrap_dim(dim, tensors[0].dim() + 1);
  return at::cat_out(result, get_stack_inputs(tensors, dim), dim);
}

static inline Tensor & sparse_transpose_(Tensor & self, int64_t dim0, int64_t dim1) {
  int64_t ndimI = self._indices().size(0);
  if (dim0 >= ndimI || dim1 >= ndimI) {
    runtime_error(
        "sparse transpose_: transposed dimensions must be sparse ",
        "Got nDimI: %llu, d0: %llu, d1: %llu",
        (long long)ndimI, (long long)dim0, (long long)dim1);
  }

  auto indices = self._indices();
  auto row0 = indices.select(0, dim0);
  auto row1 = indices.select(0, dim1);

  // swap row0 and row1
  auto tmp = at::zeros_like(row0);
  tmp.copy_(row0);
  row0.copy_(row1);
  row1.copy_(tmp);

  std::vector<int64_t> sizes(self.sizes());
  std::swap(sizes[dim0], sizes[dim1]);

  return self.sparse_raw_resize_(sizes, -1, -1);
}

Tensor & transpose_(Tensor & self, int64_t dim0, int64_t dim1) {
  auto ndims = self.dim();
  dim0 = maybe_wrap_dim(dim0, ndims);
  dim1 = maybe_wrap_dim(dim1, ndims);
  if (dim0 == dim1) {
    return self;
  }

  if (self.is_sparse()) {
    return sparse_transpose_(self, dim0, dim1);
  }

  std::vector<int64_t> strides(self.strides());
  std::vector<int64_t> sizes(self.sizes());
  std::swap(strides[dim0], strides[dim1]);
  std::swap(sizes[dim0], sizes[dim1]);
  return self.as_strided_(sizes, strides);
}

Tensor & t_(Tensor & self) {
  if (self.ndimension() != 2) {
    runtime_error("t_() expects a 2D tensor, but self is %llu",
                  (long long)self.ndimension());
  }
  return self.transpose_(0, 1);
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
inferSqueezeGeometry(const Tensor& tensor, int64_t dim) {
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

std::tuple<std::vector<int64_t>, std::vector<int64_t> >
inferUnsqueezeGeometry(const Tensor& tensor, int64_t dim) {
  if (tensor.numel() == 0) {
    throw std::runtime_error("cannot unsqueeze empty tensor");
  }

  std::vector<int64_t> sizes(tensor.sizes());
  std::vector<int64_t> strides(tensor.strides());
  int64_t new_stride = dim >= tensor.dim() ? 1 : sizes[dim] * strides[dim];
  sizes.insert(sizes.begin() + dim, 1);
  strides.insert(strides.begin() + dim, new_stride);

  return std::make_tuple(sizes, strides);
}

Tensor squeeze(const Tensor& self) {
  auto g = inferSqueezeGeometry(self);
  return self.as_strided(std::get<0>(g), std::get<1>(g));
}

Tensor squeeze(const Tensor& self, int64_t dim) {
  int64_t dims = self.dim();
  dim = maybe_wrap_dim(dim, dims);

  if (dims == 0 || self.sizes()[dim] != 1) {
    return self.as_strided(self.sizes().vec(), self.strides().vec());
  }
  auto g = inferSqueezeGeometry(self, dim);
  return self.as_strided(std::get<0>(g), std::get<1>(g));
}

Tensor & squeeze_(Tensor& self) {
  auto g = inferSqueezeGeometry(self);
  return self.as_strided_(std::get<0>(g), std::get<1>(g));
}

Tensor & squeeze_(Tensor& self, int64_t dim) {
  int64_t dims = self.dim();
  dim = maybe_wrap_dim(dim, self.dim());

  if (dims == 0 || self.sizes()[dim] != 1) {
    return self.as_strided_(self.sizes().vec(), self.strides().vec());
  }
  auto g = inferSqueezeGeometry(self, dim);
  return self.as_strided_(std::get<0>(g), std::get<1>(g));
}

// _unsafe_view() differs from view() in that the returned tensor isn't treated
// as a view for the purposes of automatic differentiation. (It's not listed in
// VIEW_FUNCTIONS in gen_autograd.py).  It's only safe to use if the `self` tensor
// is temporary. For example, the viewed tensor here is discarded immediately
// after viewing:
//
//  res = at::_unsafe_view(a + b, size);
//
// This is a hack because in-place operations on tensors treated like views
// can be much more expensive than the same operations on non-view tensors.
Tensor _unsafe_view(const Tensor& self, IntList size) {
  return self.view(size);
}

Tensor unsqueeze(const Tensor& self, int64_t dim) {
  dim = maybe_wrap_dim(dim, self.dim() + 1);

  auto g = inferUnsqueezeGeometry(self, dim);
  return self.as_strided(std::get<0>(g), std::get<1>(g));
}

Tensor & unsqueeze_(Tensor& self, int64_t dim) {
  dim = maybe_wrap_dim(dim, self.dim() + 1);

  auto g = inferUnsqueezeGeometry(self, dim);
  return self.as_strided_(std::get<0>(g), std::get<1>(g));
}

Tensor view_as(const Tensor& self, const Tensor& other) {
  return self.view(other.sizes());
}

}
}
