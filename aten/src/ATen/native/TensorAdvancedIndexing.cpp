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
#include <ATen/MemoryOverlap.h>
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
DEFINE_DISPATCH(index_fill_stub);
DEFINE_DISPATCH(index_put_stub);
DEFINE_DISPATCH(index_put_accum_stub);
DEFINE_DISPATCH(masked_fill_stub);
REGISTER_NO_CPU_DISPATCH(index_put_accum_stub, index_put_accum_fn);
DEFINE_DISPATCH(masked_select_serial_stub);
DEFINE_DISPATCH(masked_select_stub);
DEFINE_DISPATCH(masked_scatter_stub);

DEFINE_DISPATCH(gather_stub);
DEFINE_DISPATCH(scatter_stub);
DEFINE_DISPATCH(scatter_fill_stub);
DEFINE_DISPATCH(scatter_add_stub);
DEFINE_DISPATCH(scatter_reduce_stub);
DEFINE_DISPATCH(scatter_scalar_reduce_stub);

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

static ptrdiff_t dataOffset(const Tensor& tensor, ptrdiff_t linearIndex) {
  auto size = tensor.sizes();
  auto stride = tensor.strides();
  int nDim = tensor.dim();
  ptrdiff_t dataOffset = 0;
  for (int i = nDim - 1; i >= 0; i--) {
    dataOffset += (linearIndex % size[i]) * stride[i];
    linearIndex /= size[i];
  }
  return dataOffset;
}

static inline int64_t wrapLinearIndex(int64_t linearIndex, int64_t numel) {
  return linearIndex < 0 ? linearIndex + numel : linearIndex;
}

static inline void checkLinearIndex(int64_t linearIndex, int64_t numel) {
  TORCH_CHECK(linearIndex < numel && linearIndex >= -numel, "out of range: ", linearIndex, " out of ", numel);
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

static AdvancedIndex make_info(Tensor self, const torch::List<c10::optional<at::Tensor>>& orig) {
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
  TORCH_CHECK(value.scalar_type() == info.src.scalar_type(),
              "Index put requires the source and destination dtypes match, "
              "got ", info.src.scalar_type(), " for the destination "
              "and ", value.scalar_type(), " for the source.");
  TensorIteratorConfig config;
  // info.src is restrided by restride_src with 0 strided dimensions
  config.set_check_mem_overlap(false);
  config.resize_outputs(false);
  config.check_all_same_dtype(false);
  config.add_output(info.src);
  config.add_input(value);
  for (auto& index : info.indices) {
    config.add_input(index);
  }
  return config.build();
}

static TensorIterator make_index_iterator(const AdvancedIndex& info) {
  TensorIteratorConfig config;
  config.set_check_mem_overlap(false)
        .check_all_same_dtype(false)
        .declare_static_dtype_and_device(info.src.scalar_type(), info.src.device())
        .add_output(Tensor())
        .add_input(info.src);
  for (auto& index : info.indices) {
    config.add_input(index);
  }
  return config.build();
}

static TensorIterator make_index_out_iterator(const AdvancedIndex& info, Tensor& result) {
  TensorIteratorConfig config;
  // info.src is a restrided view of result
  config.set_check_mem_overlap(false)
        .check_all_same_dtype(false)
        .add_output(result)
        .add_input(info.src);
  for (auto& index : info.indices) {
    config.add_input(index);
  }
  return config.build();
}

Tensor index(const Tensor & self, const torch::List<c10::optional<Tensor>>& indices) {
  TORCH_CHECK_INDEX(indices.size() <= (size_t)self.dim(), "too many indices for tensor of dimension ", self.dim(), " (got ", indices.size(), ")");

  auto info = make_info(self, indices);
  auto iter = make_index_iterator(info);
  index_stub(iter.device_type(), iter, info.indexed_sizes, info.indexed_strides);
  return iter.output();
}

Tensor quantized_index(const Tensor & self, const torch::List<c10::optional<Tensor>>& indices) {
  TORCH_INTERNAL_ASSERT(
      self.qscheme() == c10::kPerTensorAffine ||
      self.qscheme() == c10::kPerTensorSymmetric,
      "Indexing is only supported for per-Tensor quantized Tensors.");

  // For now, this is a naive implementation which does dq -> index -> q.
  // TODO(future PR): improve performance by removing the copies.
  const auto& self_dq = self.dequantize();

  TORCH_CHECK_INDEX(indices.size() <= (size_t)self.dim(), "too many indices for tensor of dimension ", self.dim(), " (got ", indices.size(), ")");

  auto info = make_info(self_dq, indices);
  auto iter = make_index_iterator(info);
  index_stub(iter.device_type(), iter, info.indexed_sizes, info.indexed_strides);
  at::Tensor res = iter.output();

  return at::quantize_per_tensor(
      res, self.q_scale(), self.q_zero_point(), self.scalar_type());
}

Tensor& index_out(Tensor& result, const Tensor & self, const torch::List<c10::optional<Tensor>>& indices) {
  TORCH_CHECK_INDEX(indices.size() <= (size_t)self.dim(), "too many indices for tensor of dimension ", self.dim(), " (got ", indices.size(), ")");
  at::assert_no_internal_overlap(result);
  at::assert_no_overlap(result, self);
  for (const c10::optional<Tensor>& index: indices) {
    if (index.has_value()) {
      at::assert_no_overlap(result, *index);
    }
  }

  auto info = make_info(self, indices);
  auto iter = make_index_out_iterator(info, result);
  index_stub(iter.device_type(), iter, info.indexed_sizes, info.indexed_strides);
  return result;
}

Tensor index_put(const Tensor & self, const torch::List<c10::optional<Tensor>>& indices, const Tensor & value, bool accumulate) {
  return self.clone(at::MemoryFormat::Preserve).index_put_(indices, value, accumulate);
}

Tensor & _index_put_impl_(Tensor & self, const torch::List<c10::optional<Tensor>>& indices, const Tensor & value, const bool accumulate, const bool unsafe) {
  TORCH_CHECK_INDEX(indices.size() <= (size_t)self.dim(), "too many indices for tensor of dimension ", self.dim(), " (got ", indices.size(), ")");
  if (at::has_internal_overlap(self) == MemOverlap::YES) {
    TORCH_WARN(
      "Use of index_put_ on expanded tensors is deprecated. "
      "Please clone() the tensor before performing this operation. "
      "This also applies to advanced indexing e.g. tensor[indices] = tensor");
  }
  at::assert_no_overlap(self, value);
  for (const c10::optional<Tensor>& index: indices) {
    if (index.has_value()) {
      at::assert_no_overlap(self, *index);
    }
  }

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


Tensor & index_put_(Tensor & self, const torch::List<c10::optional<Tensor>>& indices, const Tensor & value, const bool accumulate) {
  return at::_index_put_impl_(self, indices, value, accumulate, /*unsafe=*/false);
}

Tensor & index_copy_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  // See note [Writing Nondeterministic Operations]
  // Nondeterministic when index contains duplicate entries
  at::globalContext().alertNotDeterministic("index_copy");
  dim = maybe_wrap_dim(dim, self.dim());

  TORCH_CHECK_INDEX(index.dim() < 2, "index_copy_(): Index should have dimension 1 or 0 (got ", index.dim(), ")");
  at::assert_no_internal_overlap(self);
  at::assert_no_overlap(self, index);
  at::assert_no_overlap(self, source);

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
  TORCH_CHECK(index.scalar_type() == ScalarType::Long || index.scalar_type() == ScalarType::Int,
          "index_add_(): Expected dtype int32/int64 for index");
  TORCH_CHECK(self.scalar_type() == source.scalar_type(),
              "index_add_(): self and source must have the same scalar type");
  TORCH_CHECK(dim == 0 || dim < source.dim(),
              "index_add_(): Indexing dim ", dim, " is out of bounds of tensor");
  TORCH_CHECK(numel == (source.dim() == 0 ? 1 : source.size(dim)),
              "index_add_(): Number of indices should be equal to self.size(dim)");

  at::assert_no_internal_overlap(self);
  at::assert_no_overlap(self, index);
  at::assert_no_overlap(self, source);

  auto index_contig = index.contiguous();

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

    AT_DISPATCH_INDEX_TYPES(index.scalar_type(), "index_add_cpu_", [&] () {
      auto index_data = index_contig.data_ptr<index_t>();
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
    });
  }
  else {
    TORCH_CHECK(source.dim() <= 1, "source.dim() (", source.dim(), ") must one or zero for given self.dim() (", self.dim(), ")");

    // explicitly capture all required variables to work around windows build
    // TODO: fix this when windows can correctly capture variables in nested lambda
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX(self.scalar_type(), "index_add_", [&self, &source, &dim, &index_contig, &numel] {
      auto self_stride = self.dim() == 0 ? 1 : self.stride(dim);
      auto source_stride = source.dim() == 0 ? 1 : source.stride(dim);
      // TODO: Maybe TensorAccessor can beused here?
      auto* self_ptr = self.data_ptr<scalar_t>();
      auto* source_ptr = source.data_ptr<scalar_t>();
      AT_DISPATCH_INDEX_TYPES(index_contig.scalar_type(), "index_add_cpu_",
        [&index_contig, &numel, &self, &self_ptr, &self_stride, &source_ptr, &source_stride] {
        auto index_data = index_contig.data_ptr<index_t>();
        for (auto i = 0; i < numel; i++) {
            auto self_i = index_data[i];
            TORCH_CHECK_INDEX((self_i >= 0) && (self_i < self.numel()), "index out of range in self");
            scalar_t *self_ip = self_ptr + self_i * self_stride;
            *self_ip += *(source_ptr + i * source_stride);
        }
      });
    });
  }
  return self;
}

Tensor index_add(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  return self.clone(at::MemoryFormat::Preserve).index_add_(dim, index, source);
}

// Check that indices fall within dimension array size
// Avoid redispatch call to min/max
template <typename IndexType>
static void check_indexarray_range(
    const IndexType* indices,
    int64_t n,
    IndexType indexing_axis_dim) {
  for (auto i = 0; i < n; ++i) {
    auto idx = indices[i];
    TORCH_CHECK(
        0 <= idx && idx < indexing_axis_dim,
        "INDICES element is out of DATA bounds, id=",
        idx,
        " axis_dim=",
        indexing_axis_dim);
  }
}

Tensor & index_select_out_cpu_dim1_(
    Tensor & result_contig, const Tensor & self, const Tensor & index_contig) {

  auto self_contig = self.contiguous();
  const caffe2::TypeMeta dataType = self_contig.dtype();
  size_t item_bytesize = dataType.itemsize();

  auto out = static_cast<char*>(result_contig.data_ptr());

  auto src_base = static_cast<const char*>(self_contig.data_ptr());

  auto self_sizes = self_contig.sizes();
  auto outer_dims_product = c10::size_to_dim_(1, self_sizes);
  auto block_size = c10::size_from_dim_(2, self_sizes);
  auto block_bytesize = block_size * item_bytesize;

  auto src_indexing_axis_dim = self_sizes[1];
  auto src_batch_bytesize = self_sizes[1] * block_bytesize;
  auto N = index_contig.numel();

  auto gathered_batch_bytesize = N * block_bytesize;

  AT_DISPATCH_INDEX_TYPES(
    index_contig.scalar_type(), "batch_index_select_compute", [&]() {

      const auto* idxs = index_contig.data_ptr<index_t>();
      check_indexarray_range<index_t>(idxs, N, src_indexing_axis_dim);

      // Special-case single-float copy for efficiency
      if (self.scalar_type() == ScalarType::Float && block_size == 1) {
        for (auto batch = 0; batch < outer_dims_product; ++batch) {
          const float* src_floats =
              (const float*)(src_base + batch * src_batch_bytesize);
          float* dst_floats = (float*)(out + batch * gathered_batch_bytesize);

          for (auto i = 0; i < N; ++i) {
            auto idx = idxs[i];
            if (idx < 0) {
              idx = idx + src_indexing_axis_dim;
            }
            dst_floats[i] = src_floats[idx];
          }
        }
      } else {
        // outer_dims_product specifies how many times we repeat inner dimensions,
        // so we just iterate over it to cover all outer dimensions.
        for (auto batch = 0; batch < outer_dims_product; ++batch) {
          for (auto i = 0; i < N; ++i) {
            auto idx = idxs[i];
            if (idx < 0) {
              idx = idx + src_indexing_axis_dim;
            }

            auto src = src_base + batch * src_batch_bytesize + idx * block_bytesize;
            auto dst = out + batch * gathered_batch_bytesize + i * block_bytesize;
            memcpy(dst, src, block_bytesize);
          }
        }
      }
  });
  return result_contig;
}

Tensor & index_select_out_cpu_(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index) {
  dim = maybe_wrap_dim(dim, self.dim());

  auto numel = index.numel();
  TORCH_CHECK_INDEX(index.dim() <= 1, "index_select(): Index is supposed to be a vector");
  TORCH_CHECK(index.scalar_type() == ScalarType::Long || index.scalar_type() == ScalarType::Int, "index_select(): Expected dtype int32 or int64 for index");
  TORCH_CHECK(self.scalar_type() == result.scalar_type(),
              "index_select(): self and result must have the same scalar type");
  TORCH_CHECK(dim == 0 || dim < self.dim(),
              "index_select(): Indexing dim ", dim, " is out of bounds of tensor");
  at::assert_no_internal_overlap(result);
  at::assert_no_overlap(result, self);
  at::assert_no_overlap(result, index);

  auto result_size = self.sizes().vec();
  if (self.dim() > 0) {
    result_size[dim] = numel;
  }
  result.resize_(result_size);

  auto index_contig = index.contiguous();

  if (self.dim() > 1) {
    if (numel == 0 || self.numel() == 0) {
      return result;
    }

    if (dim == 1 && result.is_contiguous()) {
      // fast pass
      return index_select_out_cpu_dim1_(result, self, index_contig);
    }

    auto selfSlice = self.select(dim, 0);
    auto resultSlice = result.select(dim, 0);
    auto selfSlice_data = selfSlice.data_ptr();
    auto resultSlice_data = resultSlice.data_ptr();
    auto self_stride_bytes = self.stride(dim) * elementSize(self.scalar_type());
    auto result_stride_bytes = result.stride(dim) * elementSize(result.scalar_type());
    auto self_dim_size = self.size(dim);
    auto slice_size = selfSlice.numel();

    auto iter = TensorIteratorConfig()
      .check_all_same_dtype(false)
      .resize_outputs(false)
      .add_output(resultSlice)
      .add_input(selfSlice)
      .build();

    auto grain_size = at::internal::GRAIN_SIZE;
    auto outer_loop =
      // explicitly capture all required variables to work around windows build
      // TODO: fix this when windows can correctly capture variables in nested lambda
      [&index_contig, &iter, &self_dim_size, &selfSlice_data, &self_stride_bytes, &resultSlice_data,
        &result_stride_bytes](int64_t start, int64_t end) {
      auto sub_iter = TensorIterator(iter);
      AT_DISPATCH_INDEX_TYPES(index_contig.scalar_type(), "index_select_out_cpu_",
        [&index_contig, &start, &end, &sub_iter, &self_dim_size, &selfSlice_data, &self_stride_bytes,
          &resultSlice_data, &result_stride_bytes] () {
        auto index_data = index_contig.data_ptr<index_t>();
        for (int64_t i = start; i < end; i++) {
          auto self_i = index_data[i];
          TORCH_CHECK_INDEX((self_i >= 0) && (self_i < self_dim_size), "index out of range in self");
          auto self_data = static_cast<char*>(selfSlice_data) + self_i * self_stride_bytes;
          auto result_data = static_cast<char*>(resultSlice_data) + i * result_stride_bytes;
          sub_iter.unsafe_replace_operand(0, result_data);
          sub_iter.unsafe_replace_operand(1, self_data);
          copy_stub(sub_iter.device_type(), sub_iter, false);
        };
      });
    };

    // parallel on inner loop in case the slice is large enough;
    // otherwise parallel on outer loop
    if (slice_size >= grain_size) {
      outer_loop(0, numel);
    } else {
      // use a fast loop when self and result are contiguous and of the same data type
      if (iter.is_contiguous() && self.scalar_type() == result.scalar_type()) {
        auto slice_size_bytes = slice_size * elementSize(self.scalar_type());
        // explicitly capture all required variables to work around windows build
        // TODO: fix this when windows can correctly capture variables in nested lambda
        at::parallel_for(0, numel, grain_size / slice_size,
          [&index_contig, &slice_size_bytes, &self_dim_size, &selfSlice_data,
            &self_stride_bytes, &resultSlice_data, &result_stride_bytes](int64_t start, int64_t end) {
          AT_DISPATCH_INDEX_TYPES(index_contig.scalar_type(), "index_select_out_cpu_",
            [&index_contig, &slice_size_bytes, &self_dim_size, &selfSlice_data,
              &self_stride_bytes, &resultSlice_data, &result_stride_bytes, &start, &end] () {
            auto index_data = index_contig.data_ptr<index_t>();
            for (int64_t i = start; i < end; i++) {
              auto self_i = index_data[i];
              TORCH_CHECK_INDEX((self_i >= 0) && (self_i < self_dim_size), "index out of range in self");
              auto self_data = static_cast<char*>(selfSlice_data) + self_i * self_stride_bytes;
              auto result_data = static_cast<char*>(resultSlice_data) + i * result_stride_bytes;
              memcpy(result_data, self_data, slice_size_bytes);
            }
          });
        });
      } else {
        at::parallel_for(0, numel, grain_size / slice_size, outer_loop);
      }
    }
  } else {
    TORCH_CHECK(result.dim() <= 1, "result.dim() (", result.dim(), ") must one or zero for given self.dim() (", self.dim(), ")");
    // explicitly capture all required variables to work around windows build
    // TODO: fix this when windows can correctly capture variables in nested lambda
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(at::ScalarType::Bool, self.scalar_type(), "index_select",
      [&index_contig, &self, &result, &dim, &numel] {
      auto self_stride = self.dim() == 0 ? 1 : self.stride(dim);
      auto result_stride = result.dim() == 0 ? 1 : result.stride(dim);

      auto self_data_ptr = self.data_ptr<scalar_t>();
      auto result_data_ptr = result.data_ptr<scalar_t>();
      auto self_numel = self.numel();
      AT_DISPATCH_INDEX_TYPES(index_contig.scalar_type(), "index_select_out_cpu_",
        [&index_contig, &numel, &self_numel, &self_data_ptr, &self_stride, &result_data_ptr, &result_stride] {
        auto index_data = index_contig.data_ptr<index_t>();
        for (auto i = 0; i < numel; i++) {
          auto self_i = index_data[i];
          TORCH_CHECK_INDEX((self_i >= 0) && (self_i < self_numel), "index out of range in self");
          scalar_t *self_ip = self_data_ptr + self_i * self_stride;
          *(result_data_ptr + i * result_stride) = *self_ip;
        }
      });
    });
  }

  return result;
}

Tensor index_select_cpu_(const Tensor & self, int64_t dim, const Tensor & index) {
  Tensor result = at::empty({0}, self.options());
  return index_select_out_cpu_(result, self, dim, index);
}

Tensor index_select_backward(const Tensor& grad, IntArrayRef self_sizes, int64_t dim, const Tensor& index) {
  return at::zeros(self_sizes, grad.options()).index_add_(dim, index, grad);
}

Tensor & index_fill_(Tensor & self, int64_t dim, const Tensor & index, const Scalar& source) {
  at::NoNamesGuard guard;

  TORCH_CHECK_INDEX(
    index.scalar_type() == ScalarType::Long,
    "index_fill_(): Expected dtype int64 for index.");

  at::assert_no_overlap(self, index);
  if (at::has_internal_overlap(self) == at::MemOverlap::YES) {
    TORCH_WARN(
      "Use of index_fill_ on expanded tensors is deprecated. "
      "Please clone() the tensor before performing this operation. "
      "This also applies to advanced indexing e.g. tensor[mask] = scalar");
  }

  if (!self.is_complex() && source.isComplex()) {
    TORCH_CHECK(false, "index_fill_(): Converting complex Scalar to non-complex type is forbidden");
  }

  // Handle the case when `self` is 0-dim
  Tensor self_nonzero_dim = (self.dim() == 0) ? self.unsqueeze(-1) : self;

  dim = at::maybe_wrap_dim(dim, self_nonzero_dim);
  TORCH_CHECK(index.dim() <= 1, "Index has to be a vector/scalar");

  // Prepare `index` for TensorIterator.
  // It is restrided to be broadcastable over `self` in TensorIterator.
  auto index_sizes = std::vector<int64_t>(self_nonzero_dim.dim(), 1);
  auto index_strides = std::vector<int64_t>(self_nonzero_dim.dim(), 0);
  index_sizes[dim] = index.numel();
  index_strides[dim] = (index.dim() > 0) ? index.stride(0) : 1; // `index` is 1d or scalar
  auto index_restrided = index.as_strided(
    index_sizes, index_strides);

  // Prepare `self` for TensorIterator.
  // Restride `self` to not advance in dimension `dim`.
  // We do not use squash_dim here because `index` will
  // need to advance in this dimension.
  // Note that self_sizes[dim] is set to index.numel().
  // This is done so that self_sizes[dim] and index_sizes[dim]
  // match as required by TensorIterator (input shape should
  // strictly broadcast over output shape, i.e.
  // output.shape[i] >= input.shape[i] for i in range(dims)).
  auto self_sizes = self_nonzero_dim.sizes().vec();
  auto self_strides = self_nonzero_dim.strides().vec();
  self_sizes[dim] = index.numel();
  self_strides[dim] = 0;
  auto self_restrided = self_nonzero_dim.as_strided(self_sizes, self_strides);

  auto iter = TensorIteratorConfig()
    // We do not check for overlap because `self` is restrided
    // with zero stride. Zero strides trigger memory overlap assert
    // within TensorIterator.
    .set_check_mem_overlap(false)
    .check_all_same_dtype(false)
    .resize_outputs(false)
    .add_output(self_restrided)
    .add_input(index_restrided)
    .build();

  auto self_dim_size = (self_nonzero_dim.sizes())[dim];
  auto self_dim_stride = (self_nonzero_dim.strides())[dim];
  index_fill_stub(
    iter.device_type(),
    iter,
    dim,
    self_dim_size,
    self_dim_stride,
    source);

  return self;
}

Tensor & index_fill_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  TORCH_CHECK(source.dim() == 0, "index_fill_ only supports a 0-dimensional value tensor, but got tensor "
      "with ", source.dim(), " dimension(s).");
  return self.index_fill_(dim, index, source.item());
}

Tensor index_fill(const Tensor & self, int64_t dim, const Tensor & index, const Scalar& source) {
  return self.clone(at::MemoryFormat::Preserve).index_fill_(dim, index, source);
}

Tensor index_fill(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  return self.clone(at::MemoryFormat::Preserve).index_fill_(dim, index, source);
}

Tensor & gather_out_cpu_cuda(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) {
  result.resize_(index.sizes());
  at::assert_no_internal_overlap(result);
  at::assert_no_overlap(result, self);
  at::assert_no_partial_overlap(result, index);
  gather_stub(result.device().type(), result, self, dim, index);
  return result;
}

Tensor gather(const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) {
  Tensor result = at::empty({0}, self.options());
  return gather_out_cpu_cuda(result, self, dim, index, sparse_grad);
}

Tensor gather_backward(const Tensor& grad, const Tensor& self, int64_t dim, const Tensor& index, bool sparse_grad) {
  if (sparse_grad) {
    return at::_gather_sparse_backward(self, dim, index, grad);
  }
  return at::zeros(self.sizes(), grad.options()).scatter_add_(dim, index, grad);
}

Tensor & scatter_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  TORCH_CHECK_INDEX(index.scalar_type() == ScalarType::Long,
                    "scatter_(): Expected dtype int64 for index.");
  at::assert_no_internal_overlap(self);
  at::assert_no_overlap(self, source);
  at::assert_no_overlap(self, index);
  scatter_stub(self.device().type(), self, dim, index, source);
  return self;
}

Tensor & scatter_fill_(Tensor & self, int64_t dim, const Tensor & index, const Scalar& source) {
  TORCH_CHECK_INDEX(index.scalar_type() == ScalarType::Long,
                    "scatter_(): Expected dtype int64 for index.");
  at::assert_no_internal_overlap(self);
  at::assert_no_overlap(self, index);
  scatter_fill_stub(self.device().type(), self, dim, index, source);
  return self;
}

SCATTER_GATHER_OP get_operator_enum(const std::string& reduce) {
  if (reduce == "add") {
    return SCATTER_GATHER_OP::REDUCE_ADD;
  }
  else if (reduce == "multiply") {
    return SCATTER_GATHER_OP::REDUCE_MULTIPLY;
  }
  else {
    TORCH_CHECK(false,
                "reduce argument must be either add or multiply.");
  }
}

Tensor& scatter_scalar_reduce_(Tensor& self, const int64_t dim, const Tensor& index,
                                   const Scalar& value, const std::string reduce) {
  TORCH_CHECK_INDEX(index.scalar_type() == ScalarType::Long,
                    "scatter_(): Expected dtype int64 for index.");
  TORCH_CHECK(at::isFloatingType(self.scalar_type()) || at::isComplexType(self.scalar_type()),
              "scatter_(): Expected floating or complex type for self.");
  at::assert_no_internal_overlap(self);
  at::assert_no_overlap(self, index);
  SCATTER_GATHER_OP op = get_operator_enum(reduce);
  scatter_scalar_reduce_stub(self.device().type(), self, dim, index, value, op);
  return self;
}

Tensor & scatter_reduce_(Tensor & self, const int64_t dim, const Tensor & index,
                      const Tensor & src, const std::string reduce) {
  TORCH_CHECK_INDEX(index.scalar_type() == ScalarType::Long,
                    "scatter_(): Expected dtype int64 for index");
  TORCH_CHECK(at::isFloatingType(self.scalar_type()) || at::isComplexType(self.scalar_type()),
              "scatter_(): Expected floating or complex type for self.");
  at::assert_no_internal_overlap(self);
  at::assert_no_overlap(self, index);
  at::assert_no_overlap(self, src);
  SCATTER_GATHER_OP op = get_operator_enum(reduce);
  scatter_reduce_stub(self.device().type(), self, dim, index, src, op);
  return self;
}

Tensor scatter(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  return self.clone(at::MemoryFormat::Preserve).scatter_(dim, index, source);
}

Tensor scatter(const Tensor & self, int64_t dim, const Tensor & index, const Scalar& source) {
  return self.clone(at::MemoryFormat::Preserve).scatter_(dim, index, source);
}

Tensor & scatter_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) {
  TORCH_CHECK_INDEX(index.scalar_type() == ScalarType::Long,
                    "scatter_(): Expected dtype int64 for index.");
  at::assert_no_internal_overlap(self);
  at::assert_no_overlap(self, index);
  at::assert_no_overlap(self, src);
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

static Tensor & masked_fill_impl_cpu(Tensor & self, const Tensor & mask, const Scalar& value) {
  NoNamesGuard guard;
  if (mask.dtype() == ScalarType::Byte) {
    TORCH_WARN("masked_fill_ received a mask with dtype torch.uint8, this behavior is now deprecated," \
            "please use a mask with dtype torch.bool instead.");
  }

  if (at::has_internal_overlap(self) == MemOverlap::YES) {
    TORCH_WARN(
      "Use of masked_fill_ on expanded tensors is deprecated. "
      "Please clone() the tensor before performing this operation. "
      "This also applies to advanced indexing e.g. tensor[mask] = scalar");
  }
  at::assert_no_partial_overlap(self, mask);

  auto iter = TensorIteratorConfig()
    .set_check_mem_overlap(false)  // deprecated, but not a hard error
    .check_all_same_dtype(false)
    .resize_outputs(false)
    .add_output(self)
    .add_input(mask)
    .build();

  masked_fill_stub(iter.device_type(), iter, value);
  return self;
}

Tensor & masked_fill__cpu(Tensor& self, const Tensor & mask, const Scalar& value) {
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

Tensor masked_fill(const Tensor & self, const Tensor & mask, const Scalar& source) {
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

static Tensor & masked_select_out_impl_cpu(Tensor & result, const Tensor & self, const Tensor & mask) {
  NoNamesGuard guard;

  TORCH_CHECK(mask.scalar_type() == ScalarType::Byte || mask.scalar_type() == ScalarType::Bool,
              "masked_select: expected BoolTensor or ByteTensor for mask");
  TORCH_CHECK(self.scalar_type() == result.scalar_type(),
              "masked_select(): self and result must have the same scalar type");

  at::assert_no_internal_overlap(result);
  at::assert_no_overlap(result, self);
  at::assert_no_overlap(result, mask);

  if (mask.dtype() == at::ScalarType::Byte) {
    TORCH_WARN("masked_select received a mask with dtype torch.uint8, this behavior is now deprecated," \
            "please use a mask with dtype torch.bool instead.");
  }

  Tensor _mask, _self;
  std::tie(_mask, _self) = expand_outplace(mask, self);

  auto shape = _self.sizes();
  int64_t numel = _mask.sum().item().toLong();
  result.resize_({numel});
  if (numel == 0) {
    return result;
  }

  // Create strided view of result before feeding into TensorIterator
  auto strides = DimVector(shape.size(), 0);
  auto orig_stride = result.strides()[0];
  auto result_strided = result.as_strided(shape, strides);

  // serial kernel
  // serial kernel requires that src is traversed in its logical order. However, TensorIterator might
  // have reordered dimensions so that src would be traversed in its physical order, producing wrong
  // answers. A sufficient condition that no reorder happened is that both _self and _mask is contiguous.
  // If it is not satisfied, use parallel kernel that handles permutations correctly
  bool use_serial_kernel = (self.numel() < at::internal::GRAIN_SIZE || at::get_num_threads() == 1 ) &&
  _self.is_contiguous() && _mask.is_contiguous();
  if (use_serial_kernel) {
    auto iter = TensorIteratorConfig()
      .set_check_mem_overlap(false)  // result is intenionally zero-strided above
      .check_all_same_dtype(false)
      .resize_outputs(false)
      .add_output(result_strided)
      .add_input(_self)
      .add_input(_mask)
      .build();

    masked_select_serial_stub(iter.device_type(), iter, orig_stride);
    return result;
  }

  // Use a prefix sum to record the output locations of the masked elements,
  // so as to parallel with TensorIterator.
  auto mask_long = at::empty(shape, self.options().dtype(at::kLong)).copy_(_mask);
  auto mask_prefix_sum = at::empty(shape, self.options().dtype(at::kLong));
  auto mask_long_data = mask_long.data_ptr<int64_t>();
  auto mask_prefix_sum_data = mask_prefix_sum.data_ptr<int64_t>();
  // TODO: Here can only use std::partial_sum for C++14,
  // use std::exclusive_scan when PyTorch upgrades to C++17, which have better peformance.
  // std::exclusive_scan(mask_long_data, mask_long_data + mask_long.numel(), mask_prefix_sum_data, 0);
  std::partial_sum(mask_long_data, mask_long_data + mask_long.numel(), mask_prefix_sum_data);

  auto iter = TensorIteratorConfig()
    .set_check_mem_overlap(false)  // result is intenionally zero-strided above
    .check_all_same_dtype(false)
    .resize_outputs(false)
    .add_output(result_strided)
    .add_input(_self)
    .add_input(_mask)
    .add_input(mask_prefix_sum)
    .build();

  masked_select_stub(iter.device_type(), iter, orig_stride);
  return result;
}

Tensor & masked_select_out_cpu(Tensor & result, const Tensor & self, const Tensor & mask) {
  namedinference::compute_broadcast_outnames(self, mask);
  return masked_select_out_impl_cpu(result, self, mask);
}

Tensor masked_select_cpu(const Tensor & self, const Tensor & mask) {
  Tensor result = at::empty({0}, self.options());
  return masked_select_out_cpu(result, self, mask);
}

Tensor masked_select_backward(const Tensor& grad, const Tensor& input, const Tensor& mask) {
  // The following could just be written as `zeros_like(input).masked_scatter(mask, grad)`.
  // However, as an optimization, we call the in-place variant of masked_scatter.
  // Unfortunately, that doesn't allow for the broadcasting of the LHS, so we need
  // to explicitly broadcast here (the out-of-place variant of masked_scatter
  // implicitly handles broadcasting).
  auto result = at::zeros_like(
      input.expand(at::infer_size(input.sizes(), mask.sizes())), at::MemoryFormat::Preserve);
  return result.masked_scatter_(mask, grad);
}

void take_out_cpu_template(
    Tensor& output,
    Tensor const& input,
    Tensor const& index)
{
    TORCH_CHECK(output.device().type() == at::kCPU, "device type of output (", output.device().type(), ") is not CPU");
    TORCH_CHECK(input.device().type() == at::kCPU, "device type of input (", input.device().type(), ") is not CPU");
    TORCH_CHECK(index.device().type() == at::kCPU, "device type of index (", index.device().type(), ") is not CPU");

    TORCH_CHECK(output.layout() == Layout::Strided, "take() only supports strided layout, got layout: ",
            output.layout(), " on output tensor");
    TORCH_CHECK(input.layout() == Layout::Strided, "take() only supports strided layout, got layout: ",
            input.layout(), " on input tensor");
    TORCH_CHECK(index.layout() == Layout::Strided, "take() only supports strided layout, got layout: ",
            index.layout(), " on index tensor");

    TORCH_CHECK(output.scalar_type() == input.scalar_type(), "output and input scalar type must match.",
            "But got different types: ", output.scalar_type(), " and ", input.scalar_type());
    TORCH_CHECK(index.scalar_type() == kLong, "index must be an int64 tensor");

    output.resize_(index.sizes());
    auto output_contiguous = output.contiguous();
    auto index_continuous = index.contiguous();
    bool is_contiguous = input.is_contiguous();
    auto input_size = input.numel();
    at::assert_no_internal_overlap(output);
    at::assert_no_partial_overlap(output, index);
    at::assert_no_overlap(output, input);

    AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Bool, at::ScalarType::Half, input.scalar_type(), "take_cpu", [&] {
        auto output_data = output_contiguous.data_ptr<scalar_t>();
        auto input_data = input.data_ptr<scalar_t>();
        auto index_data = index.data_ptr<int64_t>();

        // Exceptions must not be thrown across parallel sections, so we
        // record the position of the invalid index and throw the exception after the
        // loop.
        std::atomic<int64_t> invalidIdxPos(-1);

        at::parallel_for(0, index.numel(), at::internal::GRAIN_SIZE,
            [&](int64_t start, int64_t end) {
            for (auto i = start; i < end; i++) {
                int64_t idx = index_data[i];
                if (idx < input_size && idx >= -input_size) {
                    idx = wrapLinearIndex(idx, input_size);
                    if (is_contiguous) {
                        output_data[i] = input_data[idx];
                    } else {
                        output_data[i] = input_data[dataOffset(input, idx)];
                    }
                } else {
                    int64_t tmp = -1;
                    invalidIdxPos.compare_exchange_strong(tmp, i);
                }
            }
        });

        if (invalidIdxPos >= 0) {
            checkLinearIndex(index_data[invalidIdxPos], input_size);
        }
    });
}

Tensor take_cpu(const Tensor& self, const Tensor& index) {
    auto output = at::empty(index.sizes(), self.options());
    take_out_cpu_template(output, self, index);
    return output;
}

Tensor& take_out_cpu(Tensor& out, const Tensor& self, const Tensor& index) {
    take_out_cpu_template(out, self, index);
    return out;
}

Tensor take_backward(const Tensor& grad, const Tensor& input, const Tensor& index) {
  return at::zeros_like(input).put_(index, grad, true);
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

Tensor & masked_scatter__cpu(Tensor& self, const Tensor & mask, const Tensor & source) {
  at::assert_no_internal_overlap(self);
  TORCH_CHECK(
      self.scalar_type() == source.scalar_type(),
      "masked_scatter: expected self and source to have same dtypes but got",
      self.scalar_type(),
      " and ",
      source.scalar_type());

  TORCH_CHECK(self.device().type() == at::kCPU, "device type of self (", self.device().type(), ") is not CPU");
  TORCH_CHECK(mask.device().type() == at::kCPU, "device type of mask (", mask.device().type(), ") is not CPU");
  TORCH_CHECK(source.device().type() == at::kCPU, "device type of source (", source.device().type(), ") is not CPU");

  Tensor b_mask;
  std::tie(b_mask) = expand_inplace(self, mask, "masked_scatter_");

  if (b_mask.dtype() == ScalarType::Byte) {
    TORCH_WARN("masked_scatter_ received a mask with dtype torch.uint8, this behavior is now deprecated," \
            "please use a mask with dtype torch.bool instead.");
  }

  auto src_cont = source.contiguous();

  auto iter = TensorIteratorConfig()
      .set_check_mem_overlap(false)
      .check_all_same_dtype(false)
      .resize_outputs(false)
      .add_output(self)
      .add_input(b_mask)
      .build();

  masked_scatter_stub(iter.device_type(), iter, src_cont);
  return self;
}

}} // at::native
