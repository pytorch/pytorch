// Indexing tensors by by tensors
//
// This corresponds to "advanced indexing" in NumPy. The two operations are:
//
//  index(Tensor self, indices) -> Tensor
//  index_put_(Tensor self, indices, value, accumulate=false)
//
// The index is a TensorList containg kLong, kBool or kByte tensors or nulls. Byte
// tensors (boolean masks) are expanded to long tensors via nonzero(). Null
// tensors signify that the dimension is not indexed.
//
// All indexes are broadcast together and iterated as *one*. From NumPy:
//
// result[i_1, ..., i_M] == x[ind_1[i_1, ..., i_M], ind_2[i_1, ..., i_M],
//                           ..., ind_N[i_1, ..., i_M]]
//
// Note 1: ByteTensors expand to index as many dimensions as there are in the
// mask.
//
// Note 2: The behavior is more complicated when the index tensors are not all
// adjacent (e.g. x[[0, 1], :, [2, 3]]). In this case, self and the index
// tensors are transposed to the front: x.transpose(1, 2)[[0, 1], [2, 3]]
//
// The code contains two implementations of indexing. The more efficient
// implementation treats indexing like an elementwise operation over the
// tensors `result`, `x`, `ind_1`, `ind_2`, etc. This implementation does
// not work for index_put_ with accumulate=True. The other implementation
// combines the indexed tensors into a single linear index that is used
// with Tensor.put_. This is used for index_put_ with accumulate=True.
//
// The more efficient implementation takes the following steps for the
// above operation:
//
// 1) Broadcast ind_1, ind_2, ind_3 together to a common shape
// 2) Record x.stride(i) for each indexed dimension `i`
// 3) Replace the indexed subspace of `x` with the shape of the corresponding
//    subspace of `result` but with stride 0
// 4) Add dimensions of size 1 to the index tensors (ind_1, ind_2, etc.) so
//    that their shape is compatible with the result shape
//
// The CPU or CUDA kernel then computes element-wise over the broadcasted
// and restrided result, x, ind_1,  ind_2, etc.:
//
//   result[...] = *(&x[...] +
//                   ind_1[...] * x.stride(1) +
//                   ind_2[...] * x.stride(2) +
//                   ...)
//
// where & and * represent the C-style address-of and indirection operations.

#include <ATen/native/TensorAdvancedIndexing.h>
#include <ATen/native/IndexingUtils.h>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/ExpandUtils.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/Copy.h>
#include <ATen/Parallel.h>

#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>

namespace at { namespace native {

DEFINE_DISPATCH(index_stub);
DEFINE_DISPATCH(index_put_stub);
DEFINE_DISPATCH(index_put_accum_stub);
DEFINE_DISPATCH(masked_fill_stub);
REGISTER_NO_CPU_DISPATCH(index_put_accum_stub, index_put_accum_fn);

DEFINE_DISPATCH(gather_stub);
DEFINE_DISPATCH(scatter_stub);
DEFINE_DISPATCH(scatter_fill_stub);
DEFINE_DISPATCH(scatter_add_stub);

static bool all_strides_match(TensorList tensors) {
  TORCH_CHECK(tensors.size() >= 1);
  auto strides = tensors[0].strides();
  for (auto& tensor : tensors.slice(1)) {
    if (!strides.equals(tensor.strides())) {
      return false;
    }
  }
  return true;
}

static std::string shapes_as_str(TensorList tensors) {
  std::ostringstream os;
  bool first = true;
  for (auto& tensor : tensors) {
    if (tensor.defined()) {
      if (!first) {
        os << ", ";
      }
      os << tensor.sizes();
      first = false;
    }
  }
  return os.str();
}

// Replace indexed dimensions in src with stride 0 and the size of the result tensor.
// The offset in these dimensions is computed by the kernel using the index tensor's
// values and the stride of src. The new shape is not meaningful. It's used to make
// the shape compatible with the result tensor.
static Tensor restride_src(const Tensor& src, int64_t dims_before, int64_t dims_indexed,
                           IntArrayRef replacement_shape) {
  auto shape = DimVector(src.sizes());
  auto strides = DimVector(src.strides());
  int64_t end = dims_before + dims_indexed;
  shape.erase(shape.begin() + dims_before, shape.begin() + end);
  strides.erase(strides.begin() + dims_before, strides.begin() + end);
  shape.insert(shape.begin() + dims_before, replacement_shape.begin(), replacement_shape.end());
  strides.insert(strides.begin() + dims_before, replacement_shape.size(), 0);
  return src.as_strided(shape, strides);
}

// Add dimensions of size 1 to an index tensor so that it can be broadcast to the result
// shape and iterated over element-wise like the result tensor and the restrided src.
static Tensor reshape_indexer(const Tensor& index, int64_t dims_before, int64_t dims_after) {
  auto orig_shape = index.sizes();
  auto shape = DimVector();
  shape.append(dims_before, 1);
  shape.append(orig_shape.begin(), orig_shape.end());
  shape.append(dims_after, 1);
  return index.reshape(shape);
}

AdvancedIndex::AdvancedIndex(const Tensor& src, TensorList indices_list)
{
  int64_t element_size_bytes = src.element_size();
  int64_t dims_before = 0, dims_after = 0, dims_indexed = 0;
  IntArrayRef replacement_shape;
  for (size_t dim = 0; dim < indices_list.size(); dim++) {
    if (!indices_list[dim].defined()) {
      if (dims_indexed == 0) {
        dims_before++;
      } else {
        dims_after++;
      }
    } else {
      dims_indexed++;
      replacement_shape = indices_list[dim].sizes();
      indexed_sizes.push_back(src.size(dim));
      indexed_strides.push_back(src.stride(dim) * element_size_bytes);
    }
  }

  // Check if the indexed subspace contains a dim of size 0, but the replacement
  // shape does not. This implies that an index is out of bounds, because there
  // is no number that's a valid index for an empty tensor. Normally, out of
  // bounds is handled in the indexing kernel, but this case fails earlier in
  // restride_src with an unhelpful error message.
  if (std::find(indexed_sizes.begin(), indexed_sizes.end(), 0) != indexed_sizes.end() &&
      std::find(replacement_shape.begin(), replacement_shape.end(), 0) == replacement_shape.end()) {
    TORCH_CHECK_INDEX(false, "index is out of bounds for dimension with size 0");
  }

  this->dims_before = dims_before;
  this->dims_after = dims_after;
  this->src = restride_src(src, dims_before, dims_indexed, replacement_shape);

  for (auto& index : indices_list) {
    if (index.defined()) {
      indices.push_back(reshape_indexer(index, dims_before, dims_after));
    }
  }

  // For CUDA tensors, force all index tensors to have the same striding to
  // simplify the CUDA kernel.
  if (indices.size() >= 2 && this->src.device().type() == kCUDA) {
    if (!all_strides_match(indices)) {
      for (size_t i = 0; i < indices.size(); i++) {
        indices[i] = indices[i].contiguous();
      }
    }
  }
}

static AdvancedIndex make_info(Tensor self, TensorList orig) {
  checkIndexTensorTypes(orig);
  // first expand BoolTensor (masks) or ByteTensor (masks) into 1 or more LongTensors
  auto indices = expandTensors(self, orig);
  // next broadcast all index tensors together
  try {
    indices = expand_outplace(indices);
  } catch (std::exception& e) {
    TORCH_CHECK_INDEX(false, "shape mismatch: indexing tensors could not be broadcast together"
                   " with shapes ", shapes_as_str(indices));
  }
  // add missing null Tensors so that it matches self.dim()
  while (indices.size() < (size_t)self.dim()) {
    indices.emplace_back();
  }
  // if the non-null indices are not all adjacent, transpose self and indices
  // together so that they're adjacent at the front
  if (!hasContiguousSubspace(indices)) {
    std::tie(self, indices) = transposeToFront(self, indices);
  }
  // Ensure indices are on the same device as self
  for (size_t i = 0; i < indices.size(); i++) {
    if (indices[i].defined() && indices[i].device() != self.device()) {
      indices[i] = indices[i].to(self.device());
    }
  }
  return AdvancedIndex(self, indices);
}

static TensorIterator make_index_put_iterator(const AdvancedIndex& info, const Tensor& value) {
  TORCH_CHECK(is_expandable_to(value.sizes(), info.src.sizes()), "shape mismatch: value tensor of shape ", value.sizes(),
             " cannot be broadcast to indexing result of shape ", info.src.sizes());
  auto iter = TensorIterator();
  iter.dont_compute_common_dtype();
  iter.dont_resize_outputs();
  iter.add_output(info.src);
  iter.add_input(value, info.src.device(), info.src.scalar_type());
  for (auto& index : info.indices) {
    iter.add_input(index);
  }
  iter.build();
  return iter;
}

static TensorIterator make_index_iterator(const AdvancedIndex& info) {
  auto iter = TensorIterator();
  iter.dont_compute_common_dtype();
  iter.add_output(Tensor(), info.src.device(), info.src.scalar_type());
  iter.add_input(info.src);
  for (auto& index : info.indices) {
    iter.add_input(index);
  }
  iter.build();
  return iter;
}

Tensor index(const Tensor & self, TensorList indices) {
  TORCH_CHECK_INDEX(indices.size() <= (size_t)self.dim(), "too many indices for tensor of dimension ", self.dim(), " (got ", indices.size(), ")");

  auto info = make_info(self, indices);
  auto iter = make_index_iterator(info);
  index_stub(iter.device_type(), iter, info.indexed_sizes, info.indexed_strides);
  return iter.output();
}

Tensor index_put(const Tensor & self, TensorList indices, const Tensor & value, bool accumulate) {
  return self.clone(at::MemoryFormat::Preserve).index_put_(indices, value, accumulate);
}

Tensor & _index_put_impl_(Tensor & self, TensorList indices, const Tensor & value, const bool accumulate, const bool unsafe) {
    TORCH_CHECK_INDEX(indices.size() <= (size_t)self.dim(), "too many indices for tensor of dimension ", self.dim(), " (got ", indices.size(), ")");
  if (accumulate && self.device().type() == kCUDA) {
      TORCH_CHECK(value.device() == self.device(), "expected device ", self.device(), " but got device ",
      value.device(), " for value tensor");
      index_put_accum_stub(self.device().type(), self, indices, value, unsafe);
      return self;
  }
  auto info = make_info(self, indices);
  auto iter = make_index_put_iterator(info, value);
  index_put_stub(iter.device_type(), iter, info.indexed_sizes, info.indexed_strides, accumulate);
  return self;
}


Tensor & index_put_(Tensor & self, TensorList indices, const Tensor & value, const bool accumulate) {
  return at::_index_put_impl_(self, indices, value, accumulate, /*unsafe=*/false);
}

Tensor & index_copy_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  dim = maybe_wrap_dim(dim, self.dim());

  TORCH_CHECK_INDEX(index.dim() < 2, "index_copy_(): Index should have dimension 1 or 0 (got ", index.dim(), ")");

  int64_t numIndices = index.numel();
  if (source.dim() == 0 && numIndices != 1) {
    TORCH_CHECK_INDEX(false, "index_copy_(): When source is scalar, index should have one element (got ", numIndices, ")");
  } else if ((source.dim() != self.dim()) && (source.dim() != 0 && self.dim() != 0)) {
    TORCH_CHECK_INDEX(false, "index_copy_(): When source and destination are not scalars, their dimensionality must match. Source dimensionality (",
                   source.dim(), "), destination dimensionality (", self.dim(), ")");
  }

  TORCH_CHECK_INDEX(index.scalar_type() == ScalarType::Long, "index_copy_(): Expected LongTensor for index");

  // Check that source and destination slices have the same size
  auto selfSlicedSizes = self.sizes().vec();
  if (selfSlicedSizes.size() > 0) {
    selfSlicedSizes.erase(selfSlicedSizes.begin() + dim);
  }
  auto sourceSlicedSizes = source.sizes().vec();
  if (sourceSlicedSizes.size() > 0) {
    sourceSlicedSizes.erase(sourceSlicedSizes.begin() + dim);
  }
  if (selfSlicedSizes.size() != sourceSlicedSizes.size() ||
      !std::equal(selfSlicedSizes.begin(), selfSlicedSizes.end(),
                  sourceSlicedSizes.begin())) {
    std::stringstream ss;
    ss << "index_copy_(): Source/destination tensor must have same slice shapes. ";
    ss << "Destination slice shape: " << selfSlicedSizes << " at dimension " << dim;
    ss << " and source slice shape: " << sourceSlicedSizes << " at dimension 0.";
    TORCH_CHECK(false, ss.str());
  }
  TORCH_CHECK_INDEX(source.dim() == 0 || numIndices == source.size(dim),
          "index_copy_(): Number of indices (", numIndices, ") should be equal to source.size(dim) (", source.size(dim), ")");

  return at::_index_copy_(self, dim, index, source);
}

Tensor index_copy(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  return self.clone(at::MemoryFormat::Preserve).index_copy_(dim, index, source);
}


Tensor& index_add_cpu_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  dim = maybe_wrap_dim(dim, self.dim());

  auto numel = index.numel();
  TORCH_CHECK_INDEX(index.dim() <= 1, "index_add_(): Index is supposed to be a vector");
  TORCH_CHECK(index.scalar_type() == ScalarType::Long, "index_add_(): Expected dtype int64 for index");
  TORCH_CHECK(self.scalar_type() == source.scalar_type(),
              "index_add_(): self and source must have the same scalar type");
  TORCH_CHECK(dim == 0 || dim < source.dim(),
              "index_add_(): Indexing dim ", dim, " is out of bounds of tensor");
  TORCH_CHECK(numel == (source.dim() == 0 ? 1 : source.size(dim)),
              "index_add_(): Number of indices should be equal to self.size(dim)");

  auto index_contig = index.contiguous();
  auto index_data = index_contig.data_ptr<int64_t>();

  if (self.dim() > 1) {
    // Equivalent to:
    //   for (auto i = 0; i < numel; i++) {
    //     auto selfSlice = self.select(dim, index_data[i]);
    //     auto sourceSlice = source.select(dim, i);
    //     selfSlice.add_(sourceSlice);
    //   }
    // But much faster as this reuses the iterator from add_
    if (numel == 0) {
      return self;
    }
    auto selfSlice = self.select(dim, 0);
    auto sourceSlice = source.select(dim, 0);
    auto self_stride_bytes = self.stride(dim) * elementSize(self.scalar_type());
    auto source_stride_bytes = source.stride(dim) * elementSize(source.scalar_type());
    auto self_dim_size = self.size(dim);
    auto iter = TensorIterator::binary_op(selfSlice, selfSlice, sourceSlice);

    for (auto i = 0; i < numel; i++) {
      auto self_i = index_data[i];
      TORCH_CHECK_INDEX((self_i >= 0) && (self_i < self_dim_size), "index out of range in self");
      auto self_data = static_cast<char*>(selfSlice.data_ptr()) + self_i * self_stride_bytes;
      auto source_data = static_cast<char*>(sourceSlice.data_ptr()) + i * source_stride_bytes;
      iter.unsafe_replace_operand(0, self_data);
      iter.unsafe_replace_operand(1, self_data);
      iter.unsafe_replace_operand(2, source_data);
      add_stub(iter.device_type(), iter, 1);
    }
  }
  else {
    TORCH_CHECK(source.dim() <= 1, "source.dim() (", source.dim(), ") must one or zero for given self.dim() (", self.dim(), ")");

    AT_DISPATCH_ALL_TYPES(self.scalar_type(), "index_add_", [&] {
      auto self_stride = self.dim() == 0 ? 1 : self.stride(dim);
      auto source_stride = source.dim() == 0 ? 1 : source.stride(dim);
      for (auto i = 0; i < numel; i++) {
        auto self_i = index_data[i];
        TORCH_CHECK_INDEX((self_i >= 0) && (self_i < self.numel()), "index out of range in self");
        scalar_t *self_ip = self.data<scalar_t>() + self_i * self_stride;
        *self_ip += *(source.data<scalar_t>() + i * source_stride);
      }
    });
  }
  return self;
}

Tensor index_add(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  return self.clone(at::MemoryFormat::Preserve).index_add_(dim, index, source);
}

Tensor & index_select_out_cpu_(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index) {
  dim = maybe_wrap_dim(dim, self.dim());

  auto numel = index.numel();
  TORCH_CHECK_INDEX(index.dim() <= 1, "index_select(): Index is supposed to be a vector");
  TORCH_CHECK(index.scalar_type() == ScalarType::Long, "index_select(): Expected dtype int64 for index");
  TORCH_CHECK(self.scalar_type() == result.scalar_type(),
              "index_select(): self and result must have the same scalar type");
  TORCH_CHECK(dim == 0 || dim < self.dim(),
              "index_select(): Indexing dim ", dim, " is out of bounds of tensor");

  auto result_size = self.sizes().vec();
  if (self.dim() > 0) {
    result_size[dim] = numel;
  }
  result.resize_(result_size);

  auto index_contig = index.contiguous();
  auto index_data = index_contig.data_ptr<int64_t>();

  if (self.dim() > 1) {
    if (numel == 0 || self.numel() == 0) {
      return result;
    }

    auto selfSlice = self.select(dim, 0);
    auto resultSlice = result.select(dim, 0);
    auto selfSlice_data = selfSlice.data_ptr();
    auto resultSlice_data = resultSlice.data_ptr();
    auto self_stride_bytes = self.stride(dim) * elementSize(self.scalar_type());
    auto result_stride_bytes = result.stride(dim) * elementSize(result.scalar_type());
    auto self_dim_size = self.size(dim);
    auto slice_size = selfSlice.numel();

    auto iter = TensorIterator();
    iter.dont_compute_common_dtype();
    iter.dont_resize_outputs();
    iter.add_output(resultSlice);
    iter.add_input(selfSlice);
    iter.build();

    auto grain_size = at::internal::GRAIN_SIZE;
    auto outer_loop = [&](int64_t start, int64_t end) {
      auto sub_iter = TensorIterator(iter);
      for (int64_t i = start; i < end; i++) {
        auto self_i = index_data[i];
        TORCH_CHECK_INDEX((self_i >= 0) && (self_i < self_dim_size), "index out of range in self");
        auto self_data = static_cast<char*>(selfSlice_data) + self_i * self_stride_bytes;
        auto result_data = static_cast<char*>(resultSlice_data) + i * result_stride_bytes;
        sub_iter.unsafe_replace_operand(0, result_data);
        sub_iter.unsafe_replace_operand(1, self_data);
        copy_stub(sub_iter.device_type(), sub_iter, false);
      }
    };

    // parallel on inner loop in case the slice is large enough;
    // otherwise parallel on outer loop
    if (slice_size >= grain_size) {
      outer_loop(0, numel);
    } else {
      // use a fast loop when self and result are contiguous and of the same data type
      if (iter.is_contiguous() && self.scalar_type() == result.scalar_type()) {
        auto slice_size_bytes = slice_size * elementSize(self.scalar_type());
        at::parallel_for(0, numel, grain_size / slice_size, [&](int64_t start, int64_t end) {
          for (int64_t i = start; i < end; i++) {
            auto self_i = index_data[i];
            TORCH_CHECK_INDEX((self_i >= 0) && (self_i < self_dim_size), "index out of range in self");
            auto self_data = static_cast<char*>(selfSlice_data) + self_i * self_stride_bytes;
            auto result_data = static_cast<char*>(resultSlice_data) + i * result_stride_bytes;
            memcpy(result_data, self_data, slice_size_bytes);
          }
        });
      } else {
        at::parallel_for(0, numel, grain_size / slice_size, outer_loop);
      }
    }
  } else {
    TORCH_CHECK(result.dim() <= 1, "result.dim() (", result.dim(), ") must one or zero for given self.dim() (", self.dim(), ")");

    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Bool, self.scalar_type(), "index_select", [&] {
      auto self_stride = self.dim() == 0 ? 1 : self.stride(dim);
      auto result_stride = result.dim() == 0 ? 1 : result.stride(dim);
      for (auto i = 0; i < numel; i++) {
        auto self_i = index_data[i];
        TORCH_CHECK_INDEX((self_i >= 0) && (self_i < self.numel()), "index out of range in self");
        scalar_t *self_ip = self.data_ptr<scalar_t>() + self_i * self_stride;
        *(result.data_ptr<scalar_t>() + i * result_stride) = *self_ip;
      }
    });
  }

  return result;
}

Tensor index_select_cpu_(const Tensor & self, int64_t dim, const Tensor & index) {
  Tensor result = at::empty({0}, self.options());
  return index_select_out_cpu_(result, self, dim, index);
}

Tensor & index_fill_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  TORCH_CHECK(source.dim() == 0, "index_fill_ only supports a 0-dimensional value tensor, but got tensor "
      "with ", source.dim(), " dimension(s).");
  return self.index_fill_(dim, index, source.item());
}

Tensor index_fill(const Tensor & self, int64_t dim, const Tensor & index, Scalar source) {
  return self.clone(at::MemoryFormat::Preserve).index_fill_(dim, index, source);
}

Tensor index_fill(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  return self.clone(at::MemoryFormat::Preserve).index_fill_(dim, index, source);
}

Tensor & gather_out_cpu(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) {
  result.resize_(index.sizes());
  gather_stub(result.device().type(), result, self, dim, index);
  return result;
}

Tensor gather_cpu(const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) {
  Tensor result = at::empty({0}, self.options());
  return gather_out_cpu(result, self, dim, index, sparse_grad);
}

Tensor & scatter_cpu_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) {
  scatter_stub(self.device().type(), self, dim, index, src);
  return self;
}

Tensor & scatter_fill_cpu_(Tensor & self, int64_t dim, const Tensor & index, Scalar src) {
  scatter_fill_stub(self.device().type(), self, dim, index, src);
  return self;
}

Tensor scatter(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  return self.clone(at::MemoryFormat::Preserve).scatter_(dim, index, source);
}

Tensor scatter(const Tensor & self, int64_t dim, const Tensor & index, Scalar source) {
  return self.clone(at::MemoryFormat::Preserve).scatter_(dim, index, source);
}

Tensor & scatter_add_cpu_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) {
  scatter_add_stub(self.device().type(), self, dim, index, src);
  return self;
}

Tensor scatter_add(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  return self.clone(at::MemoryFormat::Preserve).scatter_add_(dim, index, source);
}

Tensor masked_scatter(const Tensor & self, const Tensor & mask, const Tensor & source) {
  Tensor _mask, _self;
  std::tie(_mask, _self) = expand_outplace(mask, self);
  return _self.clone(at::MemoryFormat::Contiguous).masked_scatter_(_mask, source);
}

static Tensor & masked_fill_impl_cpu(Tensor & self, const Tensor & mask, Scalar value) {
  NoNamesGuard guard;
  if (mask.dtype() == ScalarType::Byte) {
    TORCH_WARN("masked_fill_ received a mask with dtype torch.uint8, this behavior is now deprecated," \
            "please use a mask with dtype torch.bool instead.");
  }

  auto iter = TensorIterator();
  iter.dont_compute_common_dtype();
  iter.dont_resize_outputs();
  iter.add_output(self);
  iter.add_input(mask);
  iter.build();

  masked_fill_stub(iter.device_type(), iter, value);
  return self;
}

Tensor & masked_fill__cpu(Tensor& self, const Tensor & mask, Scalar value) {
  auto maybe_outnames = namedinference::broadcast_to_outnames(self, mask, "masked_fill_");

  masked_fill_impl_cpu(self, mask, value);
  namedinference::propagate_names_if_nonempty(self, maybe_outnames);
  return self;
}

Tensor & masked_fill__cpu(Tensor& self, const Tensor & mask, const Tensor & value) {
  auto maybe_outnames = namedinference::broadcast_to_outnames(self, mask, "masked_fill_");
  TORCH_CHECK(value.dim() == 0, "masked_fill_ only supports a 0-dimensional value tensor, but got tensor "
      "with ", value.dim(), " dimension(s).");

  masked_fill_impl_cpu(self, mask, value.item());
  namedinference::propagate_names_if_nonempty(self, maybe_outnames);
  return self;
}

Tensor masked_fill(const Tensor & self, const Tensor & mask, Scalar source) {
  Tensor result;
  auto maybe_outnames = namedinference::broadcast_to_outnames(mask, self, "masked_fill");
  {
    NoNamesGuard guard;
    Tensor _mask, _self;
    std::tie(_mask, _self) = expand_outplace(mask, self);
    result = _self.clone(at::MemoryFormat::Contiguous);
    result.masked_fill_(mask, source);
  }
  namedinference::propagate_names_if_nonempty(result, maybe_outnames);
  return result;
}

Tensor masked_fill(const Tensor & self, const Tensor & mask, const Tensor & source) {
  Tensor result;
  auto maybe_outnames = namedinference::broadcast_to_outnames(mask, self, "masked_fill");
  {
    NoNamesGuard guard;
    Tensor _mask, _self;
    std::tie(_mask, _self) = expand_outplace(mask, self);
    result = _self.clone(at::MemoryFormat::Contiguous);
    result.masked_fill_(mask, source);
  }
  namedinference::propagate_names_if_nonempty(result, maybe_outnames);
  return result;
}

Tensor _gather_sparse_backward(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& grad){
// special case scalar input and/or index
    if (self.ndimension() == 0) return at::_sparse_coo_tensor_unsafe(at::empty({0,grad.numel()}, index.options()), grad, self.sizes());
    if (grad.ndimension() == 0) return at::_sparse_coo_tensor_unsafe(index.view({1,1}), grad, self.sizes());
    Tensor sparse_ind = at::empty({self.ndimension(), grad.numel()}, self.options().dtype(at::kLong));
    int64_t n_above = grad.numel();
    int64_t n_below = 1;
    if (dim < 0) dim += self.ndimension();
    for (int i=0; i<self.ndimension(); i++) {
        n_above /= grad.size(i);
        if (i == dim) {
            sparse_ind[i] = index.reshape(-1);
        } else {
            sparse_ind[i] = at::arange(grad.size(i),self.options().dtype(at::kLong)).unsqueeze(1).expand({grad.size(i), n_above}).reshape(-1).repeat(n_below);
        }
        n_below *= grad.size(i);
    }
    return at::_sparse_coo_tensor_unsafe(sparse_ind, grad.reshape(-1), self.sizes());
}

std::vector<Tensor> nonzero_numpy(const Tensor& self) {
  // special case scalar for compatibility with numpy:
  //
  // >>> np.array(5).nonzero()
  // (array([0]),)
  // >>> np.array(0).nonzero()
  // (array([], dtype=int64),)

  if (self.dim() == 0) {
    return self.unsqueeze(0).nonzero().unbind(1);
  }

  return self.nonzero().unbind(1);
}

}} // at::native
