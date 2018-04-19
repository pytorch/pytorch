#include "ATen/ATen.h"
#include "ATen/Error.h"
#include "ATen/ExpandUtils.h"
#include "ATen/NativeFunctions.h"
#include "ATen/WrapDimUtils.h"
#include "ATen/optional.h"

#include <algorithm>

namespace at {
namespace native {

static void check_cat_no_zero_dim(TensorList tensors) {
  for(size_t i = 0; i < tensors.size(); ++i) {
    auto& t = tensors[i];
    if (t.dim() == 0) {
      AT_ERROR("zero-dimensional tensor (at position %zu) cannot be concatenated", i);
    }
  }
}

Tensor & cat_out(Tensor & result, TensorList tensors, int64_t dim) {
  check_cat_no_zero_dim(tensors);
  dim = legacy_cat_wrap_dim(dim, tensors);
  return at::_cat_out(result, tensors, dim);
}

Tensor cat(TensorList tensors, int64_t dim) {
  check_cat_no_zero_dim(tensors);
  dim = legacy_cat_wrap_dim(dim, tensors);
  return at::_cat(tensors, dim);
}

std::vector<Tensor> chunk(const Tensor& self, int64_t chunks, int64_t dim) {
  if (self.dim() == 0) {
    AT_ERROR("chunk expects at least a 1-dimensional tensor");
  }
  if (chunks <= 0) {
    AT_ERROR("chunk expects `chunks` to be greater than 0, got: %lld",
             (long long)chunks);
  }
  int64_t split_size = (self.size(dim) + chunks - 1) / chunks;
  // ensure this is dispatched through Tensor/Type, rather than the native function directly.
  return self.split(split_size, dim);
}

Tensor diagflat(const Tensor& self, int64_t offset) {
  return self.contiguous().view(-1).diag(offset);
}

Tensor diagonal(const Tensor& self, int64_t offset) {
  if (self.dim() != 2) {
    throw std::runtime_error("diagonal expects a 2-dimensional tensor");
  }
  return self.diag(offset);
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
    AT_ERROR("start out of range");
  }
  if (length <= 0 || start > cur_size - length) {
    AT_ERROR("length out of range");
  }
  return at::native::slice(self, dim, start, start + length, 1);
}

Tensor permute(const Tensor& self, IntList dims) {
  auto nDims = self.dim();
  if (dims.size() != (size_t)nDims) {
    AT_ERROR("number of dims don't match in permute");
  }
  auto oldSizes = self.sizes();
  auto oldStrides = self.strides();
  std::vector<int64_t> newSizes(nDims);
  std::vector<int64_t> newStrides(nDims);
  std::vector<bool> seen(nDims);
  for (int64_t i = 0; i < nDims; i++) {
    auto dim = maybe_wrap_dim(dims[i], nDims);
    if (seen[dim]) {
      AT_ERROR("repeated dim in permute");
    }
    seen[dim] = true;
    newSizes[i] = oldSizes[dim];
    newStrides[i] = oldStrides[dim];
  }
  return self.as_strided(newSizes, newStrides);
}

Tensor repeat(const Tensor& self, IntList repeats) {
  if (repeats.size() < (size_t)self.dim()) {
    AT_ERROR("Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor");
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

// Infers the size of a dim with size -1, if it exists. Also checks that new
// shape is compatible with the number of elements.
static std::vector<int64_t> infer_size(IntList shape, int64_t numel) {
  auto res = shape.vec();
  int64_t newsize = 1;
  auto infer_dim = at::optional<int64_t>();
  for (int64_t dim = 0, ndim = shape.size(); dim != ndim; dim++) {
    if (shape[dim] == -1) {
      if (infer_dim) {
        throw std::runtime_error("only one dimension can be inferred");
      }
      infer_dim = dim;
    } else if (shape[dim] >= 0) {
      newsize *= shape[dim];
    } else {
      AT_ERROR("invalid shape dimension %zd", shape[dim]);
    }
  }

  if (numel == newsize || (infer_dim && newsize > 0 && numel % newsize == 0)) {
    if (infer_dim) {
      res[*infer_dim] = numel / newsize;
    }
    if (numel == 0) {
      // Collapse zero-element shapes into one dimension because TH handles zeros
      // in sizes strangely: x.resize_(1, 0) has shape (1,). TODO: remove this
      // once we have multi-dimensional empty tensors.
      return {0};
    }
    return res;
  }

  std::ostringstream ss;
  ss << "shape '" << shape << "' is invalid for input of size " << numel;
  throw std::runtime_error(ss.str());
}

static at::optional<std::vector<int64_t>>
compute_stride(const Tensor& self, IntList newshape) {
  auto oldstride = self.strides();
  auto oldshape = self.sizes();
  if (oldshape.empty()) {
    return std::vector<int64_t>(newshape.size(), 1);
  }

  std::vector<int64_t> newstride(newshape.size());
  int64_t view_d = newshape.size() - 1;
  // stride for each subspace in the chunk
  int64_t chunk_base_stride = oldstride.back();
  // numel in current chunk
  int64_t tensor_numel = 1;
  int64_t view_numel = 1;
  for (int64_t tensor_d = oldshape.size() - 1; tensor_d >= 0; tensor_d--) {
    tensor_numel *= oldshape[tensor_d];
    // if end of tensor size chunk, check view
    if ((tensor_d == 0) ||
        (oldshape[tensor_d - 1] != 1 && oldstride[tensor_d - 1] != tensor_numel * chunk_base_stride)) {
      while (view_d >= 0 && (view_numel < tensor_numel || newshape[view_d] == 1)) {
        newstride[view_d] = view_numel * chunk_base_stride;
        view_numel *= newshape[view_d];
        view_d--;
      }
      if (view_numel != tensor_numel) {
        return {};
      }
      if (tensor_d > 0) {
        chunk_base_stride = oldstride[tensor_d - 1];
        tensor_numel = 1;
        view_numel = 1;
      }
    }
  }
  if (view_d != -1) {
    return {};
  }
  return newstride;
}

Tensor reshape(const Tensor& self, IntList proposed_shape) {
  if (self.type().is_sparse()) {
    AT_ERROR("reshape is not implemented for sparse tensors");
  }
  auto shape = infer_size(proposed_shape, self.numel());
  if (auto stride = compute_stride(self, shape)) {
    return self.as_strided(shape, *stride);
  }
  return at::_unsafe_view(self.clone(), shape);
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
  if (split_size < 0) {
    std::ostringstream ss;
    ss << "split expects split_size be non-negative, but got split_size="
       << split_size;
    throw std::runtime_error(ss.str());
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

std::vector<Tensor> split_with_sizes(const Tensor& self, IntList split_sizes, int64_t dim) {
  if (self.dim() == 0) {
    throw std::runtime_error("split_with_sizes expects at least a 1-dimensional tensor");
  }
  int64_t dim_size = self.size(dim);
  int64_t num_splits = split_sizes.size();
  std::vector<Tensor> splits(num_splits);
  int64_t start_idx = 0;
  int64_t i;

  for (i = 0; i < num_splits; ++i) {
    auto length = split_sizes[i];
    if (length < 0) {
      std::ostringstream ss;
      ss << "split_with_sizes expects split_sizes have only non-negative "
         << "entries, but got split_sizes=" << split_sizes;
      throw std::runtime_error(ss.str());
    }
    if (start_idx >= dim_size) {
      break;
    }
    splits[i] = self.narrow(dim, start_idx, length);
    start_idx += length;
  }
  if (i < num_splits || start_idx != dim_size) {
    std::ostringstream ss;
    ss << "split_with_sizes expects split_sizes to sum exactly to "
       << dim_size << " (input tensor's size at dimension " << dim << "), "
       << "but got split_sizes=" << split_sizes;
    throw std::runtime_error(ss.str());
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
    AT_ERROR(
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
    AT_ERROR("t_() expects a 2D tensor, but self is %llu",
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
