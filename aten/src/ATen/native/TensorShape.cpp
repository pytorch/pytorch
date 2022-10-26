#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/core/DimVector.h>
#include <ATen/core/functional.h>
#include <ATen/core/IListRef.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/InferSize.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/SparseTensorUtils.h>
#include <ATen/TensorOperators.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/DimVector.h>
#include <ATen/core/IListRef.h>
#include <ATen/native/Copy.h>
#include <ATen/native/NonSymbolicBC.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/TensorShape.h>
#include <ATen/native/TypeProperties.h>
#include <ATen/native/cpu/CatKernel.h>
#include <ATen/native/cpu/SerialStackImpl.h>
#include <ATen/native/cpu/StackKernel.h>
#include <ATen/quantized/QTensorImpl.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <c10/util/SmallVector.h>
#include <c10/util/accumulate.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_conj_copy_native.h>
#include <ATen/ops/_convert_indices_from_coo_to_csr.h>
#include <ATen/ops/_convert_indices_from_csr_to_coo.h>
#include <ATen/ops/_fw_primal_copy_native.h>
#include <ATen/ops/_indices_copy_native.h>
#include <ATen/ops/_make_dual.h>
#include <ATen/ops/_make_dual_copy_native.h>
#include <ATen/ops/_mkldnn_reshape.h>
#include <ATen/ops/_mkldnn_transpose.h>
#include <ATen/ops/_neg_view_copy_native.h>
#include <ATen/ops/_reshape_alias_copy_native.h>
#include <ATen/ops/_reshape_alias_native.h>
#include <ATen/ops/_reshape_from_tensor_native.h>
#include <ATen/ops/_shape_as_tensor_native.h>
#include <ATen/ops/_sparse_broadcast_to.h>
#include <ATen/ops/_sparse_broadcast_to_copy_native.h>
#include <ATen/ops/_sparse_broadcast_to_native.h>
#include <ATen/ops/_sparse_compressed_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_coo_tensor_with_dims_and_tensors.h>
#include <ATen/ops/_sparse_csc_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_csr_tensor_unsafe.h>
#include <ATen/ops/_sparse_csr_tensor_unsafe_native.h>
#include <ATen/ops/_stack_native.h>
#include <ATen/ops/_unsafe_view.h>
#include <ATen/ops/_unsafe_view_native.h>
#include <ATen/ops/_values_copy_native.h>
#include <ATen/ops/adjoint_native.h>
#include <ATen/ops/alias.h>
#include <ATen/ops/alias_copy_native.h>
#include <ATen/ops/alias_native.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/arange_native.h>
#include <ATen/ops/as_strided_copy_native.h>
#include <ATen/ops/as_strided_native.h>
#include <ATen/ops/as_strided_scatter_native.h>
#include <ATen/ops/atleast_1d.h>
#include <ATen/ops/atleast_2d.h>
#include <ATen/ops/atleast_3d.h>
#include <ATen/ops/block_diag_native.h>
#include <ATen/ops/broadcast_tensors_native.h>
#include <ATen/ops/broadcast_to_native.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/cat_meta.h>
#include <ATen/ops/cat_native.h>
#include <ATen/ops/chunk_native.h>
#include <ATen/ops/col_indices_copy_native.h>
#include <ATen/ops/column_stack_native.h>
#include <ATen/ops/concat_native.h>
#include <ATen/ops/concatenate_native.h>
#include <ATen/ops/crow_indices_copy_native.h>
#include <ATen/ops/dense_dim_native.h>
#include <ATen/ops/detach_copy_native.h>
#include <ATen/ops/detach_native.h>
#include <ATen/ops/diag.h>
#include <ATen/ops/diag_embed.h>
#include <ATen/ops/diag_embed_native.h>
#include <ATen/ops/diag_native.h>
#include <ATen/ops/diagflat_native.h>
#include <ATen/ops/diagonal.h>
#include <ATen/ops/diagonal_backward.h>
#include <ATen/ops/diagonal_backward_native.h>
#include <ATen/ops/diagonal_copy.h>
#include <ATen/ops/diagonal_copy_native.h>
#include <ATen/ops/diagonal_native.h>
#include <ATen/ops/diagonal_scatter_native.h>
#include <ATen/ops/dsplit_native.h>
#include <ATen/ops/dstack_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/empty_quantized.h>
#include <ATen/ops/expand_as_native.h>
#include <ATen/ops/expand_copy_native.h>
#include <ATen/ops/expand_native.h>
#include <ATen/ops/flatten_dense_tensors_native.h>
#include <ATen/ops/flatten_native.h>
#include <ATen/ops/from_blob.h>
#include <ATen/ops/hsplit_native.h>
#include <ATen/ops/hstack.h>
#include <ATen/ops/hstack_native.h>
#include <ATen/ops/index_select_native.h>
#include <ATen/ops/indices_copy_native.h>
#include <ATen/ops/lift_fresh_native.h>
#include <ATen/ops/lift_native.h>
#include <ATen/ops/mH_native.h>
#include <ATen/ops/mT_native.h>
#include <ATen/ops/matrix_H_native.h>
#include <ATen/ops/meshgrid_native.h>
#include <ATen/ops/moveaxis_native.h>
#include <ATen/ops/movedim.h>
#include <ATen/ops/movedim_native.h>
#include <ATen/ops/narrow.h>
#include <ATen/ops/narrow_copy.h>
#include <ATen/ops/narrow_copy_native.h>
#include <ATen/ops/narrow_native.h>
#include <ATen/ops/new_empty_native.h>
#include <ATen/ops/new_ones_native.h>
#include <ATen/ops/numpy_T_native.h>
#include <ATen/ops/permute_copy_native.h>
#include <ATen/ops/permute_native.h>
#include <ATen/ops/ravel_native.h>
#include <ATen/ops/repeat_native.h>
#include <ATen/ops/reshape_as_native.h>
#include <ATen/ops/reshape_native.h>
#include <ATen/ops/resize_native.h>
#include <ATen/ops/row_stack_native.h>
#include <ATen/ops/select.h>
#include <ATen/ops/select_backward_native.h>
#include <ATen/ops/select_copy_native.h>
#include <ATen/ops/select_native.h>
#include <ATen/ops/select_scatter_native.h>
#include <ATen/ops/set_native.h>
#include <ATen/ops/slice.h>
#include <ATen/ops/slice_backward_native.h>
#include <ATen/ops/slice_copy_native.h>
#include <ATen/ops/slice_native.h>
#include <ATen/ops/slice_scatter_native.h>
#include <ATen/ops/sparse_coo_tensor.h>
#include <ATen/ops/sparse_coo_tensor_native.h>
#include <ATen/ops/sparse_dim_native.h>
#include <ATen/ops/split_copy_native.h>
#include <ATen/ops/split_native.h>
#include <ATen/ops/split_with_sizes.h>
#include <ATen/ops/split_with_sizes_copy_native.h>
#include <ATen/ops/split_with_sizes_native.h>
#include <ATen/ops/squeeze_copy_native.h>
#include <ATen/ops/squeeze_native.h>
#include <ATen/ops/stack_native.h>
#include <ATen/ops/sub.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/sum_to_size_native.h>
#include <ATen/ops/swapaxes_native.h>
#include <ATen/ops/swapdims_native.h>
#include <ATen/ops/t_copy_native.h>
#include <ATen/ops/t_native.h>
#include <ATen/ops/tensor.h>
#include <ATen/ops/tensor_split.h>
#include <ATen/ops/tensor_split_native.h>
#include <ATen/ops/tile_native.h>
#include <ATen/ops/transpose.h>
#include <ATen/ops/transpose_copy_native.h>
#include <ATen/ops/transpose_native.h>
#include <ATen/ops/unbind.h>
#include <ATen/ops/unbind_copy_native.h>
#include <ATen/ops/unbind_native.h>
#include <ATen/ops/unflatten_dense_tensors_native.h>
#include <ATen/ops/unflatten_native.h>
#include <ATen/ops/unfold_copy_native.h>
#include <ATen/ops/unfold_native.h>
#include <ATen/ops/unsafe_chunk_native.h>
#include <ATen/ops/unsafe_split_native.h>
#include <ATen/ops/unsafe_split_with_sizes_native.h>
#include <ATen/ops/unsqueeze_copy_native.h>
#include <ATen/ops/unsqueeze_native.h>
#include <ATen/ops/values_copy_native.h>
#include <ATen/ops/view_as_complex.h>
#include <ATen/ops/view_as_complex_copy_native.h>
#include <ATen/ops/view_as_native.h>
#include <ATen/ops/view_as_real.h>
#include <ATen/ops/view_as_real_copy_native.h>
#include <ATen/ops/view_copy_native.h>
#include <ATen/ops/view_native.h>
#include <ATen/ops/vsplit_native.h>
#include <ATen/ops/vstack.h>
#include <ATen/ops/vstack_native.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#include <ATen/ops/zeros_native.h>
#endif

#include <algorithm>
#include <cstdint>
#include <vector>
#include <c10/util/StringUtil.h>

namespace at {
namespace meta {
inline void cat_check_no_zero_dim(const MaterializedITensorListRef& tensors) {
  size_t i = 0;
  for (const Tensor& t : tensors) {
    TORCH_CHECK(
        t.dim() > 0,
        "zero-dimensional tensor (at position ", i, ") cannot be concatenated");
    i++;
  }
}

inline c10::MemoryFormat cat_compute_output_memory_format(const MaterializedITensorListRef& inputs) {
  c10::optional<c10::MemoryFormat> format = c10::nullopt;
  for (const Tensor& t : inputs) {
    auto f = t.suggest_memory_format();
    if (f == c10::MemoryFormat::Contiguous) {
        return f;
    }
    if (format.has_value() && format.value() != f) {
        return c10::MemoryFormat::Contiguous;
    }
    format = f;
  }
  return format.value();
}

TORCH_PRECOMPUTE_META_FUNC(cat)(const ITensorListRef& tensors, int64_t dim) {
  // previously, size [0] tensors were the only possible empty tensors; thus, it wasn't possible
  // to cat empty tensors unless all the other tensors were 1-dimensional, so we allowed these tensors
  // to be "skipped".  We maintain this behavior for backwards compatibility, but only for this specific
  // size (i.e. other empty sizes are not skipped).
  auto materialized = tensors.materialize();

  cat_check_no_zero_dim(materialized);
  dim = at::legacy_cat_wrap_dim(dim, materialized);

  // Checking names before the actual dimensions.
  auto maybe_outnames = namedinference::compute_cat_outnames(materialized);

  TORCH_CHECK(
      materialized.size() > 0, "torch.cat(): expected a non-empty list of Tensors");

  // Look for the first valid tensor.
  size_t valid = materialized.size();
  for (const auto i : c10::irange(materialized.size())) {
    if (!at::native::cat_should_skip_tensor(materialized[i].get())) {
      valid = i;
      break;
    }
  }

  bool all_contiguous = true;
  bool all_same_dtype = true;
  bool all_same_sizes_and_stride = true;
  auto memory_format = cat_compute_output_memory_format(materialized);

  // Compute what the output dtype should be:
  const auto& result = maybe_get_output();
  auto is_out_defined = result.defined();
  auto out_dtype = at::native::result_type(tensors);

  // If the output tensor is defined, we need to take it into account
  // when computing the actual output dtype and the flags.
  if (is_out_defined) {
    // Check for type promotion, if the output tensor is defined.
    TORCH_CHECK(
        canCast(out_dtype, result.scalar_type()),
        "torch.cat(): input types can't be cast to the desired output type ",
        result.scalar_type());
    out_dtype = result.scalar_type();
    all_contiguous = result.is_contiguous(memory_format);
  }

  // Fallback 'set_output' parameters.
  // (in case we don't find a valid tensor)
  DimVector sizes {0};
  TensorOptions options = materialized[0].get().options()
      .dtype(out_dtype)
      .memory_format(memory_format);

  // If we found a valid tensor, check whether the input tensors
  // are compatible, i.e. we can execute `cat` on them.
  bool found_valid_tensor = valid < materialized.size();
  if (found_valid_tensor) {
    TORCH_CHECK(
        dim <= materialized[valid].get().dim(), "torch.cat(): dimension ", dim, "out of range");

    // Compute the output tensor size.
    // It should have the same shape as any other valid tensor,
    // except in the dimension 'dim'.
    size_t size_at_dim = 0;
    for (const auto i : c10::irange(materialized.size())) {
      const Tensor& t = materialized[i];
      all_same_dtype = all_same_dtype && out_dtype == t.scalar_type();
      if (!at::native::cat_should_skip_tensor(t)) {
        at::native::check_cat_shape_except_dim(materialized[valid], t, dim, i);
        size_at_dim += t.size(dim);
        all_contiguous = all_contiguous && t.is_contiguous(memory_format);
        all_same_sizes_and_stride = all_same_sizes_and_stride &&
            t.sizes() == materialized[valid].get().sizes() &&
            t.strides() == materialized[valid].get().strides();
      } else {
        all_contiguous = false;
      }
    }

    // Actually set the output.
    sizes = materialized[valid].get().sizes().vec();
    sizes[dim] = size_at_dim;
    options = materialized[valid].get().options()
        .dtype(out_dtype)
        .memory_format(memory_format);
  }

  set_output_raw_strided(0, sizes, {}, options, maybe_outnames);
  // Checks for overlaps between the inputs and the output tensor.
  if (is_out_defined && found_valid_tensor) {
    at::assert_no_internal_overlap(result);
    for (const Tensor& t : materialized) {
      at::assert_no_overlap(result, t);
    }
  }

  return TORCH_PRECOMPUTE_STRUCT(cat)()
      .set_dim(dim)
      .set_valid(valid)
      .set_all_contiguous(all_contiguous)
      .set_all_same_dtype(all_same_dtype)
      .set_all_same_sizes_and_stride(all_same_sizes_and_stride)
      .set_memory_format(memory_format);
}
} // namespace meta

namespace native {

DEFINE_DISPATCH(cat_serial_stub);
DEFINE_DISPATCH(stack_serial_stub);

Tensor _reshape_from_tensor(const Tensor& self, const Tensor& shape_tensor) {
  TORCH_CHECK(shape_tensor.dim() == 1);
  std::vector<int64_t> shape;
  auto accessor = shape_tensor.accessor<int64_t, 1>();
  for (const auto i : c10::irange(shape_tensor.numel())) {
    shape.push_back(accessor[i]);
  }
  return self.reshape(IntArrayRef(shape));
}

Tensor _shape_as_tensor(const Tensor& self) {
  auto options = TensorOptions(at::kLong);
  return at::tensor(self.sizes(), options);
}

Tensor& set_(Tensor& result, Storage source) {
  int64_t new_size =
      static_cast<int64_t>(source.nbytes() / result.dtype().itemsize());
  return result.set_(source, 0, new_size, {});
}

// unify with cuda implementation?  This is not done to avoid a dispatch in resize_impl_cpu_
Tensor& set_storage_cpu_(Tensor& result, Storage storage, int64_t storage_offset, IntArrayRef size, IntArrayRef stride) {
  checkSetStorage(result, storage, storage_offset, size, stride);

  result.unsafeGetTensorImpl()->set_storage_offset(storage_offset);
  at::OptionalIntArrayRef stride_opt = stride.data() != nullptr ?
                                          at::OptionalIntArrayRef(stride) : c10::nullopt;
  // We can re-use this kernel for the meta device.
  // We just need to make sure we don't actually try to resize the (null) storage.
  at::native::resize_impl_cpu_(result.unsafeGetTensorImpl(), size, stride_opt, /*resize_storage=*/!result.is_meta());
  return result;
}

Tensor& set_storage_meta__symint(Tensor& result, Storage storage, c10::SymInt storage_offset, c10::SymIntArrayRef size, c10::SymIntArrayRef stride) {
  checkSetStorage(result, storage, storage_offset, size, stride);

  c10::SymDimVector contiguous_strides;
  if (stride.data() == nullptr) {
    // TODO: dedupe this with empty() symbolic logic
    int64_t dim = size.size();
    contiguous_strides.resize(dim);
    if (dim > 0) {
      const auto last_idx = dim - 1;
      contiguous_strides.at(last_idx) = 1;
      for (auto i = last_idx - 1; i >= 0; --i) {
        // TODO: max with 1
        contiguous_strides.at(i) = contiguous_strides.at(i+1) * size.at(i+1);
      }
    }
    stride = contiguous_strides;
  }

  // Run this before storage setting so we can access numel
  result.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride, storage_offset);

  // Matches maybe_resize_storage_cpu no-numel behavior
  if (result.sym_numel() != 0) {
    // maybe_resize_storage_cpu can handle no storage exists at all but
    // that should never be the case here
    TORCH_INTERNAL_ASSERT(storage);
    TORCH_CHECK(storage.resizable(), "Trying to resize storage that is not resizable");
    // All meta data pointers are the same, so we don't have to "re" allocate
    // it.  TODO: Actually this might not quite be correct if we use special
    // pointers to track whether or not fake cuda tensors are pinned or not
    const auto itemsize = result.dtype().itemsize();
    c10::SymInt size_bytes = at::detail::computeStorageNbytes(
        size, stride, itemsize, storage_offset);
    storage.set_nbytes(size_bytes);
  }
  return result;
}

Tensor& set__symint(Tensor& result, const Tensor& storage, c10::SymInt storage_offset, c10::SymIntArrayRef size, c10::SymIntArrayRef stride) {
  TORCH_CHECK(storage.is_contiguous(), "passed in tensor to be used as storage must be contiguous");
  return result.set__symint(storage.storage(), storage_offset + storage.sym_storage_offset(), size, stride);
}

Tensor& set_tensor_(Tensor& result, const Tensor& source) {
  if (result.unsafeGetTensorImpl() != source.unsafeGetTensorImpl()) {
    return result.set_(source.storage(), source.storage_offset(), source.sizes(), source.strides());
  }
  return result;
}

// this needs to be split along CPU/CUDA lines because we don't have a consistent
// way of getting the allocator to use for a device (c10::GetAllocator is not
// the same as at::cuda::getCUDADeviceAllocator().
Tensor& set_cpu_(Tensor& result) {
  caffe2::TypeMeta dtype = result.dtype();
  Storage storage(
      Storage::use_byte_size_t(),
      0,
      c10::GetAllocator(kCPU),
      true);
  result.set_(storage, 0, {0}, {});
  TORCH_INTERNAL_ASSERT(dtype == result.dtype());
  return result;
}

// We can't re-use the cpu kernel here because we don't want to use the cpu allocator.
Tensor& set_meta_(Tensor& result) {
  caffe2::TypeMeta dtype = result.dtype();
  Storage storage(
      Storage::use_byte_size_t(),
      0,
      c10::GetAllocator(kMeta),
      true);
  result.set_(storage, 0, {0}, {});
  TORCH_INTERNAL_ASSERT(dtype == result.dtype());
  return result;
}

Tensor sparse_broadcast_to(const Tensor& self, IntArrayRef size) {
  TORCH_CHECK(self.is_sparse(), "input must be sparse tensor");
  int64_t sparse_extra_ndim = size.size() - self.dim();
  int64_t sparse_ndim = size.size() - self.dense_dim();
  TORCH_CHECK(sparse_extra_ndim >= 0, "input not broadcastable to size with smaller dimensionality");
  Tensor indices = self._indices();
  Tensor values = self._values();
  auto nnz = values.size(0);

  std::vector<int64_t> broadcast_sizes;
  std::vector<int64_t> broadcast_dense_sizes;
  std::vector<int64_t> broadcast_dims;
  std::vector<int64_t> unchanged_dims;
  broadcast_sizes.reserve(sparse_ndim);
  broadcast_dense_sizes.reserve(self.dense_dim() + 1);
  broadcast_dims.reserve(self.sparse_dim());
  unchanged_dims.reserve(self.sparse_dim());
  int64_t nnz_factor = 1;
  int64_t min_broadcast_dim = (sparse_extra_ndim > 0 ? 0: -1);
  int64_t max_unchanged_dim = -1;
  for (int64_t i=0; i<sparse_extra_ndim; i++) {
    auto d = size[i];
    nnz_factor *= d;
    broadcast_sizes.emplace_back(d);
  }
  for (int64_t i=0; i<self.sparse_dim(); i++) {
    auto d = size[sparse_extra_ndim + i];
    if (self.size(i) != d) {
      TORCH_CHECK(self.size(i) == 1,
                  "The expanded size of the tensor (",size[sparse_extra_ndim + i],") ",
                  "must match the existing size (",self.size(i),")");
      nnz_factor *= d;
      broadcast_sizes.emplace_back(d);
      if (min_broadcast_dim == -1) {
        min_broadcast_dim = sparse_extra_ndim + i;
      }
      broadcast_dims.emplace_back(i);
    } else {
      unchanged_dims.emplace_back(i);
      max_unchanged_dim = sparse_extra_ndim + i;
    }
  }
  // to_broadcast conserves is_coalesced property iff only the last
  // sparse dimensions are expaned. Possible expansion of dense
  // dimensions can be discarded as it does not affect the is_coalesce
  // property.
  bool is_coalesced = self.dim()==0 || (self.is_coalesced() && (max_unchanged_dim < min_broadcast_dim || min_broadcast_dim == -1));

  broadcast_dense_sizes.emplace_back(nnz);
  for (int64_t i=0; i<self.dense_dim(); i++) {
    broadcast_dense_sizes.emplace_back(size[sparse_extra_ndim + self.sparse_dim() + i]);
  }

  std::vector<int64_t> new_indices_size{sparse_ndim, nnz * nnz_factor};
  std::vector<int64_t> new_values_size(values.sizes().vec());
  new_values_size[0] = new_indices_size[1];

  Tensor new_values = values.expand(broadcast_dense_sizes).repeat_interleave(nnz_factor, 0);
  Tensor new_indices = indices.new_empty(new_indices_size);
  if (broadcast_sizes.size()>0) {
    // ones(broadcast_sizes).nonzero() is equivalent to
    // product(map(arange, broadcast_sizes)) but avoids creating
    // auxilary arange tensors
    Tensor broadcast_indices = at::native::new_ones(indices, broadcast_sizes).nonzero().transpose(0, 1).tile(nnz);
    new_indices.narrow(0, 0, sparse_extra_ndim).copy_(broadcast_indices.narrow(0, 0, sparse_extra_ndim));
    for (size_t i=0; i<broadcast_dims.size(); i++) {
      int64_t j=broadcast_dims[i];
      new_indices.select(0, sparse_extra_ndim + j).copy_(broadcast_indices.select(0, sparse_extra_ndim + i));
    }
  }
  for (int64_t j:unchanged_dims) {
    new_indices.select(0, sparse_extra_ndim + j).copy_(indices.select(0, j).repeat_interleave(nnz_factor));
  }
  return at::sparse_coo_tensor(new_indices, new_values, size)._coalesced_(is_coalesced);
}

Tensor broadcast_to(const Tensor& self, IntArrayRef size) {
  return self.expand(size);
}

std::vector<Tensor> broadcast_tensors(TensorList tensors) {
  return expand_outplace(tensors);
}

TORCH_IMPL_FUNC(cat_out_cpu)
(const ITensorListRef& tensors,
 int64_t dim,
 int64_t valid,
 bool all_contiguous,
 bool all_same_dtype,
 bool all_same_sizes_and_stride,
 MemoryFormat memory_format,
 const Tensor& result) {
  if (result.numel() == 0) {
    return;
  }

  auto materialized = tensors.materialize();

  // fast path for single thread when both inputs and result are contiguous and not empty
  bool use_serial_kernel = result.numel() < at::internal::GRAIN_SIZE || at::get_num_threads() == 1;
  ScalarType dtype = materialized[valid].get().scalar_type();
  bool serial_dtype = (dtype == ScalarType::Double || dtype == ScalarType::Float || dtype == ScalarType::BFloat16);
  if (use_serial_kernel && all_contiguous && all_same_dtype && serial_dtype) {
    cat_serial_stub(kCPU, result, materialized, dim);
    return;
  }

  int64_t offset = 0;
  if (all_same_sizes_and_stride && result.is_contiguous(memory_format) &&
      all_same_dtype) {
    const Tensor& source_slice = materialized[valid];
    auto slice_dim_size = source_slice.sizes()[dim];
    auto result_slice = result.narrow(dim, 0, slice_dim_size);
    auto result_slice_data = result_slice.data_ptr();
    auto result_stride_bytes = result.stride(dim) * elementSize(result.scalar_type());

    auto iter = TensorIteratorConfig()
      .set_check_mem_overlap(false)
      .resize_outputs(false)
      .add_output(result_slice)
      .add_input(source_slice)
      .enforce_safe_casting_to_output(true)
      .build();

    for (const Tensor& tensor : materialized) {
      if (cat_should_skip_tensor(tensor)) {
        continue;
      }
      auto source_data = static_cast<char*>(tensor.data_ptr());
      auto result_data = static_cast<char*>(result_slice_data) + offset * result_stride_bytes;
      iter.unsafe_replace_operand(0, result_data);
      iter.unsafe_replace_operand(1, source_data);
      copy_stub(iter.device_type(), iter, false);
      offset += slice_dim_size;
    }
  } else {
    for (const Tensor& tensor: materialized) {
      if (cat_should_skip_tensor(tensor)) {
        continue;
      }
      auto slice_dim_size = tensor.sizes()[dim];
      auto result_slice = result.narrow(dim, offset, slice_dim_size);

      auto iter = TensorIteratorConfig()
        .set_check_mem_overlap(false)  // Already checked above
        .resize_outputs(false)
        .add_output(result_slice)
        .add_input(tensor)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .enforce_safe_casting_to_output(true)
        .build();
      copy_stub(iter.device_type(), iter, false);
      offset += slice_dim_size;
    }
  }
}

Tensor& cat_out(TensorList tensors, Dimname dim, Tensor& result) {
  TORCH_CHECK(!tensors.empty(), "expected a non-empty list of Tensors");
  return at::cat_out(result, tensors, dimname_to_position(tensors[0], dim));
}

Tensor cat(TensorList tensors, Dimname dim) {
  TORCH_CHECK(!tensors.empty(), "expected a non-empty list of Tensors");
  return at::cat(tensors, dimname_to_position(tensors[0], dim));
}

// torch.concat, alias for torch.cat
Tensor& concat_out(TensorList tensors, Dimname dim, Tensor& result) {
  return at::cat_out(result, tensors, dimname_to_position(tensors[0], dim));
}

Tensor concat(TensorList tensors, Dimname dim) {
  return at::cat(tensors, dimname_to_position(tensors[0], dim));
}

Tensor & concat_out(TensorList tensors, int64_t dim, Tensor & result) {
  return at::cat_out(result, tensors, dim);
}

Tensor concat(TensorList tensors, int64_t dim) {
  return at::cat(tensors, dim);
}

// torch.concatenate, alias for torch.cat
Tensor& concatenate_out(TensorList tensors, Dimname dim, Tensor& result) {
  return at::cat_out(result, tensors, dimname_to_position(tensors[0], dim));
}

Tensor concatenate(TensorList tensors, Dimname dim) {
  return at::cat(tensors, dimname_to_position(tensors[0], dim));
}

Tensor& concatenate_out(TensorList tensors, int64_t dim, Tensor & result) {
  return at::cat_out(result, tensors, dim);
}

Tensor concatenate(TensorList tensors, int64_t dim) {
  return at::cat(tensors, dim);
}

static bool sizes_match_except(IntArrayRef s1, IntArrayRef s2, int64_t dim_except /* should already be wrapped */) {
  if (s1.size() != s2.size()) {
    return false;
  }
  for (const auto i : c10::irange(static_cast<int64_t>(s1.size()))) {
    if (i != dim_except && s1[i] != s2[i]) {
      return false;
    }
  }
  return true;
}

// Check to see if the shape of tensors is compatible
// for being concatenated along a given dimension.
static void check_cat_sparse_dims(Tensor const &t,
  int64_t pos /* used only for debug messages */,
  IntArrayRef sizes,
  int64_t wrapped,
  int64_t sparse_dim,
  int64_t dense_dim) {
    TORCH_CHECK(t.is_sparse(),
            "Concatenating sparse tensors, but a dense tensor was found at position ", pos, ".");
    TORCH_CHECK(sizes_match_except(sizes, t.sizes(), wrapped),
            "All tensors must have the same shape: ", sizes, " (except in the concatenating dimension),"
            " but found shape: ", t.sizes(), " at position ", pos, ".");
    TORCH_CHECK(t.sparse_dim() == sparse_dim && t.dense_dim() == dense_dim,
            "All tensors must have the same sparse_dim and dense_dim: ", sparse_dim, ", ", dense_dim,
            ", but tensor at position ", pos, " has ", t.sparse_dim(), ", ", t.dense_dim(), ".");
}

static Tensor cat_sparse_impl(const MaterializedITensorListRef& tensors, int64_t dim) {
  std::vector<Tensor> indices;
  std::vector<Tensor> values;
  int64_t wrapped = maybe_wrap_dim(dim, tensors[0].get().dim());
  int64_t sparse_dim = tensors[0].get().sparse_dim();
  int64_t dense_dim = tensors[0].get().dense_dim();
  IntArrayRef sizes = tensors[0].get().sizes();
  if (wrapped < sparse_dim) {
    for (const auto i : c10::irange(tensors.size())) {
      const Tensor& t = tensors[i];
      check_cat_sparse_dims(t, i, sizes, wrapped, sparse_dim, dense_dim);
      indices.push_back(t._indices());
      values.push_back(t._values());
    }
    Tensor idxs = at::cat(indices, 1);
    Tensor vals = at::cat(values, 0);

    // We now need to move the indices of each
    // input tensor up along `dim` by an appropriate amount.
    // E.g., if t1 has indices [[2,3,4],[5,6,7]],
    // and sizes [10, 7]
    // then torch.cat((t1,t1,t1),1) should have indices
    // [[2,3,4,2,3,4,2,3,4],[5,6,7,12,13,14,19,20,21]],
    // so we need to increase idxs[1][3:6] by 7
    // and idxs[1][6:9] by 14.
    int64_t col = 0;
    int64_t cumulative_offset = 0;
    for (const auto i : c10::irange(tensors.size())) {
      const Tensor& t = tensors[i];
      int64_t this_piece_size = t._nnz();
      // cumulative_offset is zero for the first piece, so
      // don't waste time doing this operation unless i > 0.
      if (i > 0) {
        idxs[wrapped].narrow(0, col, this_piece_size) += cumulative_offset;
      }
      cumulative_offset += t.size(wrapped);
      col += this_piece_size;
    }
    auto sizes_copy = sizes.vec();
    sizes_copy[wrapped] = cumulative_offset;
    return native::sparse_coo_tensor(
        idxs,
        vals,
        sizes_copy,
        optTypeMetaToScalarType(tensors[0].get().options().dtype_opt()),
        tensors[0].get().options().layout_opt(),
        tensors[0].get().options().device_opt(),
        tensors[0].get().options().pinned_memory_opt());
  }
  else {
    // Catting along a dense dimension requires us to create new values.
    // For illustration, consider the sparse 3d tensors t1 and t2,
    // given by t1 = [[[1,2],[3,4]], ... (zeros) ..., [[5,6],[7,8]]]
    // and t2 = [... (zeros) ..., [[9, 10], [11,12]], ... (zeros) ...],
    // Their concatenation along dimension 2 is:
    // [[[1,2,0,0],[3,4,0,0]], ... (zeros) ..., [[0,0,9,10],[0,0,11,12]], ... (zeros) ..., [[5,6,0,0],[7,8,0,0]]]
    //
    // Their values tensors are, respectively,
    // [[[1,2],[3,4]],[[5,6],[7,8]]] and [[[9,10],[11,12]]].
    //
    // and so the values tensor of their concatenation along dim 2 will be:
    // [[[1,2,0,0],[3,4,0,0]],[[5,6,0,0],[7,8,0,0]],[[0,0,9,10],[0,0,11,12]]]
    //
    // which we can get by taking the values tensor of each tensor, catting it with zeros of the appropriate size on the left and right,
    // and then catting all those results together.

    // The dimension in each tensor's values object that corresponds to the overall dimension along which we're catting.
    int64_t values_dim = wrapped - sparse_dim + 1;
    // The final size along the catted dimension.
    const int64_t total_size = std::accumulate(
        tensors.begin(),
        tensors.end(),
        static_cast<int64_t>(0),
        [values_dim](int64_t l, const Tensor& r) {
          return l + r._values().size(values_dim);
        });
    auto zeros_sizes = tensors[0].get()._values().sizes().vec();
    int64_t cumulative_size = 0;
    std::vector<Tensor> vals_pieces;
    std::vector<Tensor> idxs_pieces;
    for (const auto i : c10::irange(tensors.size())) {
      const Tensor& t = tensors[i];
      check_cat_sparse_dims(t, i, sizes, wrapped, sparse_dim, dense_dim);
      // dimension 0 of values corresponds to the number of values,
      // rather than to any logical dimension of the sparse tensor.
      zeros_sizes[0] = t._values().size(0);
      zeros_sizes[values_dim] = cumulative_size;
      cumulative_size += t._values().size(values_dim);
      auto z1 = at::zeros(
          zeros_sizes,
          optTypeMetaToScalarType(t._values().options().dtype_opt()),
          t._values().options().layout_opt(),
          t._values().options().device_opt(),
          t._values().options().pinned_memory_opt());
      zeros_sizes[values_dim] = total_size - cumulative_size;
      auto z2 = at::zeros(
          zeros_sizes,
          optTypeMetaToScalarType(t._values().options().dtype_opt()),
          t._values().options().layout_opt(),
          t._values().options().device_opt(),
          t._values().options().pinned_memory_opt());
      vals_pieces.push_back(at::cat({z1, t._values(), z2}, values_dim));
      idxs_pieces.push_back(t._indices());
    }
    auto sizes_copy = sizes.vec();
    sizes_copy[wrapped] = total_size;
    // This can create an uncoalesced tensor
    return native::sparse_coo_tensor(
        at::cat(idxs_pieces, 1),
        at::cat(vals_pieces),
        sizes_copy,
        optTypeMetaToScalarType(tensors[0].get().options().dtype_opt()),
        tensors[0].get().options().layout_opt(),
        tensors[0].get().options().device_opt(),
        tensors[0].get().options().pinned_memory_opt());
  }
}

Tensor cat_sparse(const ITensorListRef& tensors, int64_t dim) {
  auto materialized = tensors.materialize();
  auto maybe_outnames = namedinference::compute_cat_outnames(materialized);
  auto result = cat_sparse_impl(materialized, at::legacy_cat_wrap_dim(dim, materialized));
  namedinference::propagate_names_if_nonempty(result, maybe_outnames);
  return result;
}

Tensor block_diag(TensorList tensors) {
  Tensor result;
  if (tensors.size() == 0) {
    result = at::empty({1, 0});
    return result;
  }

  const Device& device = tensors[0].device();
  for (const auto tensor_idx : c10::irange(tensors.size())) {
    const Tensor& tensor = tensors[tensor_idx];

    TORCH_CHECK(
      tensor.device() == device,
      "torch.block_diag: input tensors must all be on the same device.",
      " Input 0 is on device ", device,
      " and input ", tensor_idx, " is on device ", tensor.device()
    );
  }

  ScalarType output_scalar_type = native::result_type(tensors);
  int64_t result_dim0 = 0;
  int64_t result_dim1 = 0;
  std::vector<Tensor> tensors_2D(tensors.size());

  // Sum the dimensions of the tensors, check tensor sizes,
  // and expand all 0-D and 1-D tensors so that everything
  // is 2-D
  for (const auto tensor_idx : c10::irange(tensors.size())) {
    const Tensor& tensor = tensors[tensor_idx];
    int64_t ndims = tensor.dim();
    TORCH_CHECK(
      ndims <= 2,
      "torch.block_diag: Input tensors must have 2 or fewer dimensions. Input ",
      tensor_idx, " has ", ndims, " dimensions"
    );

    int64_t dim0 = 1;
    int64_t dim1 = 1;

    if (ndims == 2) {
      dim0 = tensor.size(0);
      dim1 = tensor.size(1);
      tensors_2D[tensor_idx] = tensor;
    } else if (ndims == 1) {
      // Switching dim 0 to dim 1 is intentional
      dim1 = tensor.size(0);
      tensors_2D[tensor_idx] = tensor.expand({dim0, dim1});
    } else {
      tensors_2D[tensor_idx] = tensor.expand({dim0, dim1});
    }
    result_dim0 += dim0;
    result_dim1 += dim1;
  }

  result = at::zeros(
    {result_dim0, result_dim1},
    tensors[0].options().dtype(output_scalar_type)
  );

  int64_t cur_dim0 = 0;
  int64_t cur_dim1 = 0;

  // Copy each tensor into the appropriate location in the result matrix
  for (const auto& tensor : tensors_2D) {
    int64_t dim0 = tensor.size(0);
    int64_t dim1 = tensor.size(1);
    result.slice(0, cur_dim0, cur_dim0+dim0).slice(1, cur_dim1, cur_dim1+dim1).copy_(tensor);

    cur_dim0 += dim0;
    cur_dim1 += dim1;
  }

  return result;
}

std::vector<Tensor> chunk(const Tensor& self, int64_t chunks, int64_t dim) {
  TORCH_CHECK(self.dim() > 0,
           "chunk expects at least a 1-dimensional tensor");
  TORCH_CHECK(chunks > 0,
           "chunk expects `chunks` to be greater than 0, got: ", chunks);

  const auto dim_size = self.sym_size(dim);
  auto split_size = (dim_size + chunks - 1) / chunks;

  // We need to call split_with_sizes in the case where split_size and dimension size are 0, because
  // a call to split would discard the number of chunks (because we can have an arbitrary number of
  // 0-sized chunks adding up to 0).  So, call split_with_sizes with the correct number of chunks,
  // eventually we will do this for all cases.
  if (split_size == 0 && dim_size == 0) {
    std::vector<c10::SymInt> split_sizes(chunks, split_size);
    split_sizes[chunks - 1] = split_size - (split_size * chunks - dim_size);
    return self.split_with_sizes_symint(split_sizes, dim);
  } else {
    return self.split_symint(split_size, dim);
  }
}

std::vector<Tensor> tensor_split_sections_symint(const Tensor& self, c10::SymInt sym_sections, int64_t dim) {
  TORCH_CHECK(self.dim() > 0, "tensor_split expected at least a 1-dimensional tensor, but got a tensor with ", self.dim()," dims");
  int64_t dim_ = maybe_wrap_dim(dim, self.dim());
  // NB: intentional, sections specifies number of output tensors, which
  // cannot be polymorphic
  int64_t sections = sym_sections.guard_int(__FILE__, __LINE__);
  TORCH_CHECK(sections > 0, "number of sections must be larger than 0, got ", sections);
  const auto dim_size = self.sym_size(dim_);
  std::vector<Tensor> splits(sections);
  auto min_split_size = dim_size / sections;
  auto num_splits_one_extra = dim_size % sections;
  c10::SymInt start_idx = 0;
  for (const auto split_idx : c10::irange(sections)) {
    auto split_size = (num_splits_one_extra > split_idx) ? (min_split_size + 1) : min_split_size;
    splits[split_idx] = at::slice_symint(self, dim_, start_idx, start_idx + split_size);
    start_idx += split_size;
  }
  return splits;
}

template <typename T>
std::vector<Tensor> _tensor_split_indices(const Tensor& self, ArrayRef<T> indices, int64_t dim) {
  TORCH_CHECK(self.dim() > 0, "tensor_split expected at least a 1-dimensional tensor, but got a tensor with ", self.dim()," dims");
  int64_t dim_ = maybe_wrap_dim(dim, self.dim());
  int64_t num_indices = indices.size();
  std::vector<Tensor> splits(num_indices + 1);
  T start_idx(0);
  for (const auto split_idx : c10::irange(num_indices)) {
    auto end_idx = indices[split_idx];
    splits[split_idx] = at::symint::slice<T>(self, dim_, start_idx, end_idx);
    start_idx = end_idx;
  }
  splits[num_indices] = at::symint::slice<T>(self, dim_, start_idx, at::symint::size<T>(self, dim_));
  return splits;
}

std::vector<Tensor> tensor_split(const Tensor& self, IntArrayRef indices, int64_t dim) {
  return _tensor_split_indices(self, indices, dim);
}

std::vector<Tensor> tensor_split_indices_symint(const Tensor& self, SymIntArrayRef indices, int64_t dim) {
  return _tensor_split_indices(self, indices, dim);
}

std::vector<Tensor> tensor_split(const Tensor& self, const Tensor& tensor_indices_or_sections, int64_t dim) {
  TORCH_CHECK(self.dim() > 0, "tensor_split expected at least a 1-dimensional tensor, but got a tensor with ", self.dim()," dims");
  auto split_device = tensor_indices_or_sections.device();
  TORCH_CHECK(split_device == kCPU,
    "tensor_split expected tensor_indices_or_sections to be on cpu, but it's on ", split_device);
  auto split_dtype = tensor_indices_or_sections.scalar_type();
  TORCH_CHECK(split_dtype == at::kLong,
    "tensor_split expected tensor_indices_or_sections to have dtype of long, but got ", split_dtype);
  auto split_dim = tensor_indices_or_sections.dim();
  TORCH_CHECK(split_dim == 1 || split_dim == 0,
    "tensor_split expected tensor_indices_or_sections to be a zero-dimensional or one-dimensional tensor, but got a tensor with ", split_dim, " dims");

  if (split_dim == 0) {
    int64_t sections = tensor_indices_or_sections.item<int64_t>();
    return self.tensor_split(sections, dim);
  } else {
    auto indices_data = tensor_indices_or_sections.data_ptr<int64_t>();
    auto stride = tensor_indices_or_sections.stride(0);
    auto numel = tensor_indices_or_sections.numel();
    std::vector<int64_t> indices(numel);
    for (const auto offset : c10::irange(numel)) {
      // indices tensor could be non-contiguous
      indices[offset] = *(indices_data + offset * stride);
    }
    return self.tensor_split(indices, dim);
  }
}

std::vector<Tensor> unsafe_chunk(const Tensor& self, int64_t chunks, int64_t dim) {
  TORCH_CHECK(self.dim() > 0,
           "chunk expects at least a 1-dimensional tensor");
  TORCH_CHECK(chunks > 0,
           "chunk expects `chunks` to be greater than 0, got: ", chunks);

  const auto dim_size = self.size(dim);
  int64_t split_size = (dim_size + chunks - 1) / chunks;

  // See the comment above in chunk(...)
  if (split_size == 0 && dim_size == 0) {
    std::vector<int64_t> split_sizes(chunks, split_size);
    split_sizes[chunks - 1] = split_size - (split_size * chunks - dim_size);
    return self.unsafe_split_with_sizes(split_sizes, dim);
  } else {
    return self.unsafe_split(split_size, dim);
  }
}

Tensor diagflat(const Tensor& self, int64_t offset) {
  return self.contiguous().view(-1).diag(offset);
}

Tensor diagonal(const Tensor& self, int64_t offset, int64_t dim1_, int64_t dim2_) {
  int64_t nDims = self.dim();
  int64_t dim1 = maybe_wrap_dim(dim1_, nDims);
  int64_t dim2 = maybe_wrap_dim(dim2_, nDims);
  TORCH_CHECK(dim1 != dim2, "diagonal dimensions cannot be identical ", dim1_, ", ", dim2_);
  auto outnames = namedinference::compute_diagonal_outnames(self, dim1, dim2);
  NoNamesGuard no_names_guard;

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int64_t diag_size;
  int64_t storage_offset = self.storage_offset();
  // compute storage offset and size for the diagonal
  // for positive values of offset (above the main diagonal)
  // "leftmost columns" (along dim2) are dropped
  // for negative values of offset (below the main diagonal)
  // "topmost rows" (along dim1) are dropped.
  // Note that we invert +/- in the second to absorb the negative
  // sign in the offset.
  if (offset >= 0) {
    diag_size = std::max<int64_t>(std::min(self.size(dim1), self.size(dim2)-offset), 0);
  } else {
    diag_size = std::max<int64_t>(std::min(self.size(dim1)+offset, self.size(dim2)), 0);
  }

  // NumPy allows you to specify offsets "off the end"; let's just be careful not to
  // set a ridiculous storage_offset in that case (technically it shouldn't matter
  // because there are no elements in the tensor, but let's be kosher).
  if (diag_size == 0) {
    // skip
  } else if (offset >= 0) {
    storage_offset += offset * self.stride(dim2);
  } else {
    storage_offset -= offset * self.stride(dim1);
  }

  // construct new size and stride: we drop dim1 and dim2 (maximum first for not changing the index of the minimum)
  // the new ("joint") dimension is appended to the end of the shape / stride to match numpy semantics
  DimVector sizes(self.sizes().begin(), self.sizes().end());
  DimVector strides(self.strides().begin(), self.strides().end());
  sizes.erase(sizes.begin() + std::max(dim1, dim2));
  strides.erase(strides.begin() + std::max(dim1, dim2));
  sizes.erase(sizes.begin() + std::min(dim1, dim2));
  strides.erase(strides.begin() + std::min(dim1, dim2));
  sizes.push_back(diag_size);
  strides.push_back(self.stride(dim1)+self.stride(dim2));

  // return view with new parameters
  auto result = self.as_strided(sizes, strides, storage_offset);

  no_names_guard.reset();
  namedinference::propagate_names_if_nonempty(result, outnames);
  return result;
}

Tensor diagonal(const Tensor& self, Dimname outdim, Dimname dim1, Dimname dim2, int64_t offset) {
  auto result = at::diagonal(
      self,
      offset,
      dimname_to_position(self, dim1),
      dimname_to_position(self, dim2));
  // This is slower than it needs to be because there is no way to modify
  // the names of a tensor in-place right now. In the future we should consider
  // offering that functionality.
  std::vector<Dimname> new_names = result.names().vec();
  new_names[new_names.size() - 1] = outdim;
  return result.refine_names(new_names);
}

Tensor diag_embed(const Tensor& self, int64_t offset, int64_t dim1_, int64_t dim2_) {
  int64_t nDims = self.dim() + 1;
  int64_t dim1 = maybe_wrap_dim(dim1_, nDims);
  int64_t dim2 = maybe_wrap_dim(dim2_, nDims);
  TORCH_CHECK(dim1 != dim2, "diagonal dimensions cannot be identical ", dim1_, ", ", dim2_);
  int64_t new_dim_len = std::abs(offset) + self.size(-1);
  auto sizes = self.sizes().vec();
  sizes.pop_back();
  sizes.insert(sizes.begin() + std::min(dim1, dim2), new_dim_len);
  sizes.insert(sizes.begin() + std::max(dim1, dim2), new_dim_len);
  auto result = at::zeros(sizes, self.options());
  auto diag = result.diagonal(offset, dim1, dim2);
  diag.copy_(self);
  return result;
}

Tensor expand(const Tensor& self, c10::IntArrayRef size, bool /*unused*/) {
  TORCH_CHECK(size.size() >= (size_t)self.dim(),
           "expand(", self.toString(), "{", self.sizes(), "}, size=", size,
           "): the number of sizes provided (", size.size(), ") ",
           "must be greater or equal to the number of dimensions in the tensor (",
           self.dim(), ")");

  auto expandedSizesAndStrides = inferExpandGeometry_dimvector(self.sizes(), self.strides(), size);

  auto result = self.as_strided(
      expandedSizesAndStrides.sizes, expandedSizesAndStrides.strides);
  namedinference::propagate_names_for_expand(result, self);
  return result;
}

Tensor expand_as(const Tensor& self, const Tensor& other) {
  return self.expand_symint(other.sym_sizes());
}

Tensor sum_to_size(const Tensor& self, IntArrayRef size) {
  TORCH_CHECK(is_expandable_to(size, self.sizes()),
           "size {", size, "} is not expandable to size {", self.sizes(), "}.");

  return sum_to(self, size);
}

// We currently do not support per-channel quant for unfold, diagonal, expand, permute.
// TODO: Make this an aten function and replace as_strided_qtensorimpl once that is done.
Tensor make_qtensor(const Tensor& self, IntArrayRef size, IntArrayRef stride, QuantizerPtr quantizer) {
  auto result = at::detail::make_tensor<QTensorImpl>(
      c10::TensorImpl::VIEW, Storage(self.storage()), self.key_set(), self.dtype(), quantizer);
  setStrided(result, size, stride, self.storage_offset());
  return result;
}

Tensor as_strided_tensorimpl(const Tensor& self, IntArrayRef size, IntArrayRef stride, optional<int64_t> storage_offset_) {
  TORCH_INTERNAL_ASSERT(!self.is_mps(), "as_strided_tensorimpl does not work with MPS; call self.as_strided(...) instead");
  auto storage_offset = storage_offset_.value_or(self.storage_offset());
  auto result = at::detail::make_tensor<TensorImpl>(
      c10::TensorImpl::VIEW, Storage(self.storage()), self.key_set(), self.dtype());
  setStrided(result, size, stride, storage_offset);
  return result;
}

Tensor as_strided_tensorimpl_meta(const Tensor& self, IntArrayRef size, IntArrayRef stride, optional<int64_t> storage_offset_) {
  auto storage_offset = storage_offset_.value_or(self.storage_offset());
  auto result = at::detail::make_tensor<TensorImpl>(
      c10::TensorImpl::VIEW, Storage(self.storage()), self.key_set(), self.dtype());
  setStrided(result, size, stride, storage_offset);
  return result;
}

Tensor as_strided_tensorimpl_meta_symint(const Tensor& self, SymIntArrayRef sym_size, SymIntArrayRef sym_stride, optional<c10::SymInt> sym_storage_offset_) {
  auto sym_storage_offset = sym_storage_offset_.value_or(self.sym_storage_offset());
  auto result = at::detail::make_tensor<TensorImpl>(
      c10::TensorImpl::VIEW, Storage(self.storage()), self.key_set(), self.dtype());
  setStrided(result, sym_size, sym_stride, sym_storage_offset);
  return result;
}

Tensor as_strided_qtensorimpl(const Tensor& self, IntArrayRef size, IntArrayRef stride, optional<int64_t> storage_offset_) {
  auto storage_offset = storage_offset_.value_or(self.storage_offset());
  auto quantizer = get_qtensorimpl(self)->quantizer();
  TORCH_CHECK(
      quantizer->qscheme() == QScheme::PER_TENSOR_AFFINE,
      "Setting strides is possible only on uniformly quantized tensor");
  auto result = at::detail::make_tensor<QTensorImpl>(
      c10::TensorImpl::VIEW, Storage(self.storage()), self.key_set(), self.dtype(), quantizer);
  setStrided(result, size, stride, storage_offset);
  return result;
}

// This is an overloaded function similar to
// Tensor as_strided_qtensorimpl(const Tensor& self, IntArrayRef size, IntArrayRef stride, optional<int64_t> storage_offset_)
// and is currently not available through the dispatcher. The additional
// input, quantizer, is called by the select & slice methods.
// TODO: Make this function compatible with the dispatcher
Tensor as_strided_qtensorimpl(const Tensor& self, IntArrayRef size, IntArrayRef stride, optional<int64_t> storage_offset_,
  QuantizerPtr quantizer) {
  auto storage_offset = storage_offset_.value_or(self.storage_offset());
  TORCH_CHECK(
      (quantizer->qscheme() == QScheme::PER_TENSOR_AFFINE) ||
      (quantizer->qscheme() == QScheme::PER_CHANNEL_AFFINE),
      "Setting strides is possible only on uniformly or per channel quantized tensors");
  auto result = at::detail::make_tensor<QTensorImpl>(
      c10::TensorImpl::VIEW, Storage(self.storage()), self.key_set(), self.dtype(), quantizer);
  setStrided(result, size, stride, storage_offset);
  return result;
}

const Tensor &as_strided__symint(const Tensor& self, SymIntArrayRef size, SymIntArrayRef stride, optional<c10::SymInt> storage_offset_) {
  auto storage_offset = storage_offset_.value_or(self.sym_storage_offset());
  setStrided(self, size, stride, storage_offset);
  return self;
}

Tensor narrow_copy_dense(const Tensor& self, int64_t dim, int64_t start, int64_t length) {
  return self.narrow(dim, start, length).clone(at::MemoryFormat::Contiguous);
}

Tensor narrow_copy_dense_cpu(const Tensor& self, int64_t dim, int64_t start, int64_t length){
  auto output = at::empty_like(self);
  return narrow_copy_dense_cpu_out(self, dim, start, length, output);
}

Tensor narrow_copy_sparse(const Tensor& self, int64_t dim, int64_t start, int64_t length) {
  int64_t allDim = self.dim();
  int64_t end = start+length;
  TORCH_CHECK(allDim > 0, "narrow() cannot be applied to a 0-dim tensor.");
  TORCH_CHECK(dim >= 0 && dim < allDim,
    "Dimension ", dim, " out of range. Expecting 0 <= dim < ", allDim, ".");
  TORCH_CHECK(start >= 0 && length >= 0 && end <= self.size(dim),
    "Invalid range to narrow. range(start, start+length) must be a subset of range(0, ", self.size(dim), ").")
  Tensor indices = self._indices();
  int64_t sparse_dim = self.sparse_dim();

  std::vector<int64_t> new_sizes = self.sizes().vec();
  new_sizes[dim] = length;

  Tensor new_values;
  Tensor new_indices;
  if (dim < sparse_dim) {
    Tensor mask = (indices[dim] >= start).__and__((indices[dim] < end));
    new_indices = indices.masked_select(mask).view({sparse_dim, -1});
    new_indices[dim].sub_(start);
    Tensor nzIndices = mask.nonzero().view(-1);
    new_values = self._values().index_select(0, nzIndices);
  } else {
    /* This means we are narrowing on a dense dim, which is in effect just a
        regular narrow on _values() */
    new_indices = indices;
    int64_t dense_dim = dim - sparse_dim + 1;
    new_values = self._values().narrow_copy(dense_dim, start, length);
  }

  auto newTensor = at::sparse_coo_tensor(new_indices, new_values, new_sizes);
  return newTensor._coalesced_(self.is_coalesced());
}

Tensor& narrow_copy_dense_cpu_out(
  const Tensor& self, int64_t dim, int64_t start, int64_t length, Tensor& output
) {

  TORCH_CHECK(self.dim() > 0, "narrow() cannot be applied to a 0-dim tensor.");
  TORCH_CHECK(self.dtype() == output.dtype());

  auto self_contig = self.expect_contiguous();
  const auto self_sizes = self_contig->sizes();

  // wrap dim if negative and do bound check
  if (dim < 0) {
    dim = at::maybe_wrap_dim(dim, self_sizes.size());
  } else {
    TORCH_CHECK(dim < static_cast<int64_t>(self_sizes.size()));
  }

  // wrap start and do bound check
  const auto cur_size = self_sizes[dim];
  if (start != cur_size && start < 0) { // start being the end is valid, but
                                        // not a valid dim specification.
    start = at::maybe_wrap_dim(start, cur_size);
  }
  TORCH_CHECK(
      length >= 0 && start <= cur_size - length,
      "start (",
      start,
      ") + length (",
      length,
      ") exceeds dimension size (",
      cur_size,
      ").");

  // resize output
  auto output_sizes = self_sizes.vec();
  output_sizes[dim] = length;
  at::native::resize_(output, output_sizes);

  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  const int64_t unit = c10::size_from_dim_(dim + 1, self_sizes);
  const int64_t num_blocks = c10::size_to_dim_(dim, self_sizes);

  const auto itemsize = self_contig->dtype().itemsize();
  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  size_t src_nbytes = itemsize * self_contig->numel();
  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  size_t dst_nbytes = itemsize * output.numel();

  size_t src_block_size = unit * self_sizes[dim];
  size_t dst_block_size = unit * length;

  if (num_blocks == 0 || dst_block_size == 0) {
    return output;
  }

  char* src_bytes = static_cast<char*>(self_contig->data_ptr());
  char* dst_bytes = static_cast<char*>(output.data_ptr());

  size_t src_block_size_bytes = itemsize * src_block_size;
  size_t dst_block_size_bytes = itemsize * dst_block_size;
  size_t src_offset = unit * start;

  char* src_offset_bytes = src_bytes + itemsize * src_offset;
  char* dst_offset_bytes = dst_bytes;

  for (const auto i : c10::irange(num_blocks)) {
    char* local_src_offset_bytes = src_offset_bytes + i * src_block_size_bytes;
    char* local_dst_offset_bytes = dst_offset_bytes + i * dst_block_size_bytes;
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        static_cast<void*>(local_src_offset_bytes + dst_block_size_bytes) <=
        static_cast<void*>(src_bytes + src_nbytes));
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        static_cast<void*>(local_dst_offset_bytes + dst_block_size_bytes) <=
        static_cast<void*>(dst_bytes + dst_nbytes));

    memcpy(
        local_dst_offset_bytes, local_src_offset_bytes, dst_block_size_bytes);
  }
  return output;
}

Tensor narrow(const Tensor& self, int64_t dim, int64_t start, int64_t length) {
  TORCH_CHECK(self.dim() > 0, "narrow() cannot be applied to a 0-dim tensor.");
  auto cur_size = self.size(dim);
  if (start != cur_size) {  // start being the end is valid, but not a valid dim specification.
    start = maybe_wrap_dim(start, cur_size);
  }
  TORCH_CHECK(length >= 0 && start <= cur_size - length,
           "start (", start, ") + length (", length, ") exceeds dimension size (", cur_size, ").");
  return at::slice(self, dim, start, start + length, 1);
}

Tensor narrow_symint(const Tensor& self, int64_t dim, SymInt start, SymInt length) {
  TORCH_CHECK(self.dim() > 0, "narrow() cannot be applied to a 0-dim tensor.");
  auto cur_size = self.sym_size(dim);
  if (start != cur_size) {  // start being the end is valid, but not a valid dim specification.
    start = maybe_wrap_dim(start, cur_size);
  }
  TORCH_CHECK(length >= 0 && start <= cur_size - length,
           "start (", start, ") + length (", length, ") exceeds dimension size (", cur_size, ").");
  return at::slice_symint(self, dim, start, start + length, 1);
}

// This overload exists purely for XLA, because they wanted to pass in "symbolic"
// start via Tensor.
Tensor narrow_tensor_symint(const Tensor& self, int64_t dim, const Tensor& start, SymInt length) {
  TORCH_CHECK(start.dim() == 0 && isIntegralType(start.scalar_type(), /*includeBool=*/false),
              "start must be an 0-dim integral Tensor.");
  int64_t st = start.item<int64_t>();
  return at::narrow_symint(self, dim, c10::SymInt(st), length);
}

std::tuple<DimVector, DimVector, std::vector<int64_t>>
_permute_size_stride_estimation(const Tensor& self, IntArrayRef dims) {
  const auto ndim = self.dim();
  TORCH_CHECK(ndim == static_cast<int64_t>(dims.size()),
      "permute(sparse_coo): number of dimensions in the tensor input ",
      "does not match the length of the desired ordering of dimensions ",
      "i.e. input.dim() = ", ndim, " is not equal to len(dims) = ", dims.size());

  const auto is_strided_layout = self.options().layout() == at::kStrided;
  const auto old_sizes = self.sizes();
  const auto old_strides = is_strided_layout ? self.strides() : IntArrayRef{};

  auto new_sizes = DimVector(ndim);
  auto new_strides = DimVector(is_strided_layout ? ndim : 0);
  auto wrapped_dims = std::vector<int64_t>(ndim);
  std::vector<bool> seen_dims(ndim);

  for (const auto i : c10::irange(ndim)) {
    const auto d = maybe_wrap_dim(dims[i], ndim);
    TORCH_CHECK(!seen_dims[d],
        "permute(): duplicate dims are not allowed.");
    seen_dims[d] = true;
    wrapped_dims[i] = d;
    new_sizes[i] = old_sizes[d];
    if (is_strided_layout) {
      new_strides[i] = old_strides[d];
    }
  }

  return std::make_tuple(new_sizes, new_strides, wrapped_dims);
}

Tensor permute(const Tensor& self, IntArrayRef dims) {
  DimVector new_sizes, new_strides;
  std::vector<int64_t> _;
  std::tie(new_sizes, new_strides, _) = _permute_size_stride_estimation(self, dims);
  return self.as_strided(new_sizes, new_strides);
}

Tensor permute_sparse_coo(const Tensor& self, IntArrayRef dims) {
  DimVector new_sizes, _;
  std::vector<int64_t> wrapped_dims;
  std::tie(new_sizes, _, wrapped_dims) = _permute_size_stride_estimation(self, dims);

  const auto ndim = self.dim();
  const auto sparse_ndim = self.sparse_dim();
  const auto dense_ndim = self.dense_dim();

  auto dims_id_perm = std::vector<int64_t>(ndim);
  auto dims_sparse_dense_id_perm = std::vector<int64_t>(ndim);
  for (const auto i : c10::irange(ndim)) {
    dims_id_perm[i] = i;
    dims_sparse_dense_id_perm[i] = wrapped_dims[i];
  }
  std::sort(dims_sparse_dense_id_perm.begin(), dims_sparse_dense_id_perm.begin() + sparse_ndim);
  std::sort(dims_sparse_dense_id_perm.begin() + sparse_ndim, dims_sparse_dense_id_perm.end());
  TORCH_CHECK(dims_sparse_dense_id_perm == dims_id_perm,
      "permute(sparse_coo): transpositions between sparse and dense dimensions are not allowed.",
      "Only transpositions within sparse and dense dimensions are supported.");

  const auto slice = [](std::vector<int64_t> v, size_t begin, size_t len) -> decltype(v) {
    return std::vector<int64_t>{v.begin() + begin, v.begin() + begin + len};
  };

  auto old_sparse_dims = slice(dims_id_perm, 0, sparse_ndim);
  auto old_dense_dims = slice(dims_id_perm, sparse_ndim, ndim - sparse_ndim);
  auto new_sparse_dims = slice(wrapped_dims, 0, sparse_ndim);
  auto new_dense_dims = slice(wrapped_dims, sparse_ndim, ndim - sparse_ndim);

  auto old_indices = self._indices();
  auto old_values = self._values();

  const auto new_indices = (new_sparse_dims == old_sparse_dims)
    ? old_indices
    : [&]() -> Tensor {
      auto sparse_perm_tensor = at::from_blob(reinterpret_cast<void*>(new_sparse_dims.data()),
          {sparse_ndim}, old_indices.options().device(at::kCPU));
      // creates new indices. It is possible to avoid that if COO
      // is allowed to store a permutation vector.
      return old_indices.index_select(0, sparse_perm_tensor.to(self.device().type()));
    }();
  const auto new_values = (new_dense_dims == old_dense_dims)
    ? old_values
    : [&]() -> Tensor {
      auto values_perm = std::vector<int64_t>(dense_ndim + 1);
      for (const auto i : c10::irange(dense_ndim)) {
        values_perm[i + 1] = new_dense_dims[i] - sparse_ndim + 1;
      }
      return old_values.permute(values_perm);
    }();

  const auto is_coalesced = self.is_coalesced() && (dims[0] == 0);
  return _sparse_coo_tensor_with_dims_and_tensors(
      sparse_ndim, dense_ndim, new_sizes, new_indices, new_values, self.options())
    ._coalesced_(is_coalesced);
}

Tensor repeat(const Tensor& self, IntArrayRef repeats) {
  TORCH_CHECK(repeats.size() >= (size_t)self.dim(),
           "Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor");

  // Add new leading dimensions to the tensor if the
  // number of target dimensions is larger than the
  // number of source dimensions.
  int64_t num_new_dimensions = repeats.size() - self.dim();
  DimVector padded_size(num_new_dimensions, 1);
  padded_size.insert(padded_size.end(), self.sizes().begin(), self.sizes().end());
  DimVector target_size(repeats.size());
  bool zero_tensor = false;
  for(const auto idx : c10::irange(repeats.size())) {
    if (repeats[idx] == 0) {
      zero_tensor = true;
    }
    target_size[idx] = padded_size[idx] * repeats[idx];
  }

  Tensor xtensor = self.expand(padded_size);

  Tensor result;
  if (self.is_quantized()) {
    result = at::empty_quantized(target_size, self);
  } else {
    result = at::empty(target_size, self.options());
  }

  // return an empty tensor if one of the repeat dimensions is zero
  if (zero_tensor) {
    return result;
  }

  Tensor urtensor = at::alias(result);
  for (const auto i : c10::irange(xtensor.dim())) {
    // can't unfold with step 0, so make sure step is at least 1
    // (it doesn't matter what it is in that case, because the size is 0).
    auto size_i = xtensor.sizes()[i];
    urtensor = urtensor.unfold(i, size_i, std::max<int64_t>(size_i, 1));
  }

  urtensor.copy_(xtensor.expand_as(urtensor));

  return result;
}

Tensor tile(const Tensor& self, IntArrayRef reps){
  // If self.size() > len(reps), reps is promoted to self.size() by pre-pending
  // 1’s to it to keep the same behaviour as `numpy.tile`.
  // Thus for a tensor of shape (2, 3, 4, 5), a dims of (2, 2) is treated
  // as (1, 1, 2, 2).
  const int64_t size_diff = self.dim() - static_cast<int64_t>(reps.size());
  if (size_diff > 0){
    std::vector<int64_t> new_reps(size_diff, 1);
    for (const auto i : c10::irange(reps.size())) {
      new_reps.emplace_back(reps[i]);
    }
    return self.repeat(IntArrayRef(new_reps));
  }
  // `torch.tile` is equivalent to the already implemented `torch.Tensor.repeat`
  return self.repeat(reps);
}

//
// templated for ArrayRef<int64_t> and SmallVector<int64_t> use cases
//
template <typename Vec>
Tensor alias_with_sizes_and_strides(
    const Tensor& self,
    const Vec& sizes,
    const Vec& strides) {
  //caller should make sure that sizes and strides are valid for self
  //(storage is sufficient, strides are non-negative, strides and sizes array size is the same)
  Tensor self_;
  if (self.is_quantized()) {
    self_ = at::detail::make_tensor<QTensorImpl>(
      c10::TensorImpl::VIEW, Storage(self.storage()), self.key_set(), self.dtype(), get_qtensorimpl(self)->quantizer());
    auto* self_tmp_ = self_.unsafeGetTensorImpl();
    self_tmp_->set_storage_offset(self.storage_offset());
    self_tmp_->set_sizes_and_strides(sizes, strides);
  } else {
    self_ = at::detail::make_tensor<TensorImpl>(
      c10::TensorImpl::VIEW, Storage(self.storage()), self.key_set(), self.dtype());
    auto* self_tmp_ = self_.unsafeGetTensorImpl();
    self_tmp_->set_storage_offset(self.storage_offset());
    self_tmp_->set_sizes_and_strides(sizes, strides);
  }
  namedinference::propagate_names(self_, self);
  return self_;
}

Tensor reshape_symint(const Tensor& self, c10::SymIntArrayRef proposed_shape) {
  if (self.is_sparse()) {
    AT_ERROR("reshape is not implemented for sparse tensors");
  }
  c10::SymDimVector shape = infer_size_dv(proposed_shape, self.sym_numel());

  if (self.is_mkldnn()) {
    return at::_mkldnn_reshape(self, c10::asIntArrayRefSlow(shape));
  }

  // `computeStride` returns the proper strides to use if this
  // `reshape` can be just a view.
  auto stride = at::detail::computeStride(self.sym_sizes(), self.sym_strides(), shape);

  // NB: Even though we have viewable geometry and the target strides here,
  //     we do not just call `as_strided` on `self` because the backward
  //     for `as_strided` is not as efficient as that of `view` (since the
  //     former is meant to handle general cases).
  //
  //     Similarly we don't call `view` because it duplicates some of the work
  //     we've already done, and instead call our internal/private operator
  //     `_reshape_alias` that essentially does the same thing as `view` and
  //     `as_strided` without any of the extra overhead.
  if (stride.has_value()) {
    // Temporary check to revert to the old behavior/view in cases where the
    // device is not supported (e.g. for XLA the operation is not supported
    // so we use `view` instead).
    //
    // We need to do the checks here instead of in `native_functions.yaml`
    // to preserve backwards compatibility.
    if (!self.is_xla() && !self.is_lazy() && !self.is_ipu()) {
      return self._reshape_alias_symint(shape, stride.value());
    } else {
      return self.view_symint(shape);
    }
  }
  return at::_unsafe_view_symint(self.clone(at::MemoryFormat::Contiguous), shape);
}

// Duplicate of above code for non-symbolic ints. Kept for BC purposes and to
// minimize breakages.
Tensor reshape(const Tensor& self, IntArrayRef proposed_shape) {
  if (self.is_sparse()) {
    AT_ERROR("reshape is not implemented for sparse tensors");
  }
  DimVector shape = infer_size_dv(proposed_shape, self.numel());

  if (self.is_mkldnn()) {
    return at::_mkldnn_reshape(self, shape);
  }

  // `computeStride` returns the proper strides to use if this
  // `reshape` can be just a view.
  auto stride = at::detail::computeStride(self.sizes(), self.strides(), shape);

  // NB: Even though we have viewable geometry and the target strides here,
  //     we do not just call `as_strided` on `self` because the backward
  //     for `as_strided` is not as efficient as that of `view` (since the
  //     former is meant to handle general cases).
  //
  //     Similarly we don't call `view` because it duplicates some of the work
  //     we've already done, and instead call our internal/private operator
  //     `_reshape_alias` that essentially does the same thing as `view` and
  //     `as_strided` without any of the extra overhead.
  if (stride.has_value()) {
    // Temporary check to revert to the old behavior/view in cases where the
    // device is not supported (e.g. for XLA the operation is not supported
    // so we use `view` instead).
    //
    // We need to do the checks here instead of in `native_functions.yaml`
    // to preserve backwards compatibility.
    if (!self.is_xla() && !self.is_lazy() && !self.is_ipu()) {
      return self._reshape_alias(shape, stride.value());
    } else {
      return self.view(shape);
    }
  }
  return at::_unsafe_view(self.clone(at::MemoryFormat::Contiguous), shape);
}

Tensor _reshape_alias(const Tensor& self, IntArrayRef sizes, IntArrayRef strides) {
  // This is only used by `reshape` in cases where it would otherwise have dispatched
  // to `view`. This removes the overhead of calling `view` which duplicates some of
  // the work that's already been done (`infer_size_dv` and `computeStride`).

  return alias_with_sizes_and_strides(self, sizes, strides);
}

Tensor reshape_as(const Tensor& self, const Tensor& other) {
  return self.reshape(other.sizes());
}

static Tensor select_sparse(const Tensor& self, int64_t dim, int64_t index) {
  int64_t sparse_dim = self.sparse_dim();
  int64_t dense_dim = self.dense_dim();
  TORCH_INTERNAL_ASSERT(dim >= 0 && dim < sparse_dim + dense_dim);

  auto indices = self._indices();
  auto values = self._values();
  auto new_sizes = self.sizes().vec();
  new_sizes.erase(new_sizes.begin() + dim);

  if (dim < sparse_dim) {
    auto nzIndices = (indices[dim] == index).nonzero().view(-1);
    auto new_values = values.index_select(0, nzIndices);
    if (sparse_dim == 1) {
      // return dense part:
      if (new_values.size(0) == 1) {
        return new_values[0];
      } else {
        // sum promotes integral type to int64 when dtype is not specified.
        return at::sum(new_values, 0, false, new_values.scalar_type());
      }
    } else {
      auto dimIndices = (arange(
                             0,
                             sparse_dim,
                             c10::nullopt /* dtype */,
                             c10::nullopt /* layout */,
                             self.device(),
                             c10::nullopt /* pin_memory */) != dim)
                            .nonzero()
                            .view(-1);
      auto new_indices = indices.index_select(1, nzIndices).index_select(0, dimIndices);
      return _sparse_coo_tensor_with_dims_and_tensors(
            sparse_dim - 1, dense_dim, new_sizes, new_indices, new_values, self.options());
    }
  } else {
    auto new_values = values.select(dim - sparse_dim + 1, index);
    return _sparse_coo_tensor_with_dims_and_tensors(
         sparse_dim, dense_dim - 1, new_sizes, indices, new_values, self.options());
  }
}

// this is an auxiliary function, called by the select&slice methods, that
// creates a new quantizer from the given input
// is_select is true if calling function is select()
QuantizerPtr create_subtensor_quantizer(const Tensor& self, bool is_select, int64_t start,
  int64_t end, int64_t dim, int64_t step) {
  auto quantizer_prev = get_qtensorimpl(self)->quantizer();
  if (quantizer_prev->qscheme() == QScheme::PER_TENSOR_AFFINE) {
    return quantizer_prev;
  }
  QuantizerPtr quantizer;
  auto temp = static_cast<PerChannelAffineQuantizer*>(quantizer_prev.get());
  auto axis = temp->axis();
  auto scales = temp->scales();
  auto zero_points = temp->zero_points();
  if (dim == axis) {
    // Compute scales&zps for sub-tensor
    // *.select(0, start) could alternatively be replaced with *.slice(0, start, end, step), but
    // select has less overhead
    scales = is_select ? scales.select(0, start) : scales.slice(0, start, end, step);
    zero_points = is_select ? zero_points.select(0, start) : zero_points.slice(0, start, end, step);
  }
  if (scales.numel() > 1) {
    // Axis only needs to be adjusted if the calling function is select(), since select() reduces
    // the number of dimensions of the tensor by 1, and remains unchanged if calling function is slice()
    quantizer = make_per_channel_affine_quantizer(scales, zero_points, (is_select ? axis - 1 : axis),
                                                  quantizer_prev->scalar_type());
  } else {
    quantizer = make_per_tensor_affine_quantizer(scales.item().to<double>(), zero_points.item().to<int64_t>(),
                                                 quantizer_prev->scalar_type());
  }
  return quantizer;
}

Tensor select(const Tensor& self, int64_t dim, int64_t index_) {
  int64_t ndim = self.dim();
  if (ndim == 0) {
    TORCH_CHECK_INDEX(false, "select() cannot be applied to a 0-dim tensor.");
  }
  dim = maybe_wrap_dim(dim, ndim);
  auto size = self.sym_sizes()[dim];
  if (size < -index_ || size <= index_) {
    if (self.has_names() && self.names()[dim] != Dimname::wildcard()) {
      TORCH_CHECK_INDEX(false, "select(): index ", index_, " out of range for tensor of size ",
                     self.sizes(), " at dimension ", self.names()[dim]);
    }
    TORCH_CHECK_INDEX(false, "select(): index ", index_, " out of range for tensor of size ",
                   self.sizes(), " at dimension ", dim);
  }
  SymInt index = index_;
  if (index < 0) {
    index += size;
  }
  if (self.is_sparse()) {
    return select_sparse(self, dim, index.guard_int(__FILE__, __LINE__));
  }

  Tensor result;
  if (self.is_quantized()) {
    auto local_index = index.guard_int(__FILE__, __LINE__);

    DimVector sizes(self.sizes().begin(), self.sizes().end());
    DimVector strides(self.strides().begin(), self.strides().end());
    auto storage_offset = self.storage_offset() + local_index * strides[dim];
    sizes.erase(sizes.begin() + dim);
    strides.erase(strides.begin() + dim);

    auto quantizer = create_subtensor_quantizer(self, true, local_index, local_index + 1, dim, 1);
    result = as_strided_qtensorimpl(self, sizes, strides, storage_offset, quantizer);
  } else {
    std::vector<c10::SymInt> sizes(self.sym_sizes().begin(), self.sym_sizes().end());
    std::vector<c10::SymInt> strides(self.sym_strides().begin(), self.sym_strides().end());
    auto storage_offset = self.sym_storage_offset() + index * strides[dim];
    sizes.erase(sizes.begin() + dim);
    strides.erase(strides.begin() + dim);

    result = self.as_strided_symint(sizes, strides, storage_offset);
  }
  namedinference::propagate_names_except(result, self, {dim});
  return result;
}

Tensor select(const Tensor& self, Dimname dim, int64_t index) {
  return at::select(self, dimname_to_position(self, dim), index);
}

Tensor select_backward(const Tensor& grad, IntArrayRef input_sizes, int64_t dim, int64_t index) {
  auto grad_input = at::zeros(input_sizes, grad.options());
  grad_input.select(dim, index).copy_(grad);
  return grad_input;
}

Tensor index_select_sparse_cpu(const Tensor& self, int64_t dim, const Tensor& index) {
  /*
    Algorithm:
    index - a 1-D tensor of indicies with shape (n,)
    self - sparse tensor, its shape is sizes = sparse_shape + dense_shape
      indices - 2-D tensor of indices, shape is (sparse_dims, nnz)
      values - (1+len(dense_shape))-D tensor of values, shape is (nnz,) + dense_shape
    index_select(dim, index) returns a sparse tensor with the following data
      new_sizes = sizes[:dim] + (n,) + sizes[dim+1:]
      new_indices - shape is (sparse_dims, new_nnz)
      new_values - shape is (new_nnz,) + dense_shape

      if dim < len(sparse_shape):
          # Find new_indices[dim] of the output sparse tensor and
          # indices at which to select values/indices.
          # The CPP code uses (binary/in a count table) search to find matches and may
          # swap the loop order for better algorithmic complexity.
          new_dim_indices = []
          selected_dim_indices = []
          # This is a brute-force algorithms to convey the main idea.
          # The CPP code below is more efficient but more complicated.
          for i, i_idx in enumerate(indices[dim]):
              for j, j_idx in enumerate(index):
                  if i_idx == j_idx:
                      new_dim_indices.append(j)
                      selected_dim_indices.append(i)
          new_indices = indices.index_select(1, selected_dim_indices)
          new_values = values.index_select(0, selected_dim_indices)
          new_indices[dim] = new_dim_indices
      else:
          new_indices = indices
          new_values = values.index_select(dim - sparse_dim + 1, index);
    */
  const auto ndim = self.dim();
  TORCH_CHECK_INDEX(ndim, "index_select() cannot be applied to a 0-dim tensor.");
  TORCH_CHECK_INDEX(
      index.dim() == 1 && index.dtype() == at::kLong && index.options().layout() == at::kStrided,
      "index_select() argument index must be 1-D strided (non-sparse) long-tensor.");
  dim = maybe_wrap_dim(dim, ndim);
  const auto size = self.size(dim);
  const auto sparse_dim = self.sparse_dim();
  const auto dense_dim = self.dense_dim();
  const auto indices = self._indices();
  const auto values = self._values();
  const auto nnz = values.size(0);
  const auto index_len = index.size(0);
  auto res_sizes = self.sizes().vec();
  res_sizes[dim] = index_len;

  // Equivalent to t.index_select(dim, idx), but vanilla index_select is not parallel,
  // so we use gather instead.
  // We use this method to select relevant indices/values
  // from the intersection between indices[dim] and the index.
  const auto index_select = [](const Tensor& t, int64_t dim, const Tensor& idx) -> Tensor {
    const auto idx_len = idx.numel();
    auto out_shape = t.sizes().vec();
    out_shape[dim] = idx_len;
    auto idx_shape = std::vector<int64_t>(t.dim(), 1);
    idx_shape[dim] = idx_len;
    return t.gather(dim, idx.view(idx_shape).expand(out_shape));
  };

  // If indexing into sparse dimensions
  if (dim < sparse_dim) {
    // short-circuit if index is empty
    if (!index_len) {
      auto res_indices = index_select(indices, 1, index);
      res_indices[dim] = index;
      const auto res_values = index_select(values, 0, index);

      return _sparse_coo_tensor_with_dims_and_tensors(
          sparse_dim, dense_dim, res_sizes, res_indices, res_values, self.options());
    }

    const auto nneg_index = [&index, index_len, &self, size, dim]() -> Tensor {
      const auto index_contiguous = index.contiguous();
      auto nneg_index = at::empty_like(index_contiguous);
      // nneg_index = (index < 0) * (index + size) + (index >= 0) * index
      auto* ptr_index = index_contiguous.data_ptr<int64_t>();
      auto* ptr_nneg_index = nneg_index.data_ptr<int64_t>();
      at::parallel_for(0, index_len, at::internal::GRAIN_SIZE, [&](int64_t start, int64_t end) {
          const auto* src = ptr_index + start;
          auto* dst = ptr_nneg_index + start;
          for (C10_UNUSED const auto _ : c10::irange(start, end)) {
            auto idx = *src++;
            if (idx < -size || idx >= size) {
               // Mark self and dim as used if code is compiled with STRIP_ERROR_MESSAGES
              (void)dim;
              (void)self;
              TORCH_CHECK_INDEX(false,
                  "index_select(): index contains ", idx, " that is out of range for tensor of size ",
                  self.sizes(), " at dimension ", dim
              );
            }
            if (idx < 0) {
              idx += size;
            }
            *dst++ = idx;
          }
      });

      return nneg_index;
    }();

    const auto dim_indices = indices[dim].contiguous();

    // If nnz is smaller than size, then either indices[dim] or index gets sorted,
    // then this is followed by a binary search to find interesections.
    const auto get_selected_indices_small_nnz_large_size = [&]() -> std::tuple<Tensor, Tensor> {
      const auto grain_size = at::internal::GRAIN_SIZE;
      const auto n_threads_nnz = std::max<int64_t>(
          1, std::min<int64_t>((nnz + grain_size - 1) / grain_size, at::get_num_threads())
      );
      const auto n_threads_index = std::max<int64_t>(
          1, std::min<int64_t>((index_len + grain_size - 1) / grain_size, at::get_num_threads())
      );
      const auto search_in_dim_indices
        // if either dim_indices or index requires sorting, we compare
        // the cost of sort + binary search, which is comparing
        // (len(dim_indices) + len(index)) * log(len(index)) to
        // (len(dim_indices) + len(index)) * log(len(dim_indices)).
        // That simplifies to comparing len(dim_indices) to len(index).
        // Additionally, we take into consideration potential parallel
        // speedup.
        = (nnz / n_threads_nnz <= index_len / n_threads_index)
        // if self is coalesced and dim is 0, then we compare
        // index_len * log(len(dim_indices)), which is binary search into dim_indices,
        // to (len(index_len) + len(dim_indices)) * log(index_len).
        // Additionally, we take into consideration potential parallel
        // speedup.
          || (self.is_coalesced() && dim == 0
          && (index_len * std::log2(nnz) / n_threads_index
            <= (nnz / n_threads_nnz + index_len) * std::log2(index_len)))
        ? true : false;

      // src is a source of indices to binary search in sorted
      Tensor sorted, sorted_idx, src;
      std::tie(sorted, sorted_idx, src) = [
        &dim_indices, &nneg_index, &self,
        search_in_dim_indices, dim, nnz
      ](void) -> std::tuple<Tensor, Tensor, Tensor> {
        // sort dim_indices to binary search into it
        if (search_in_dim_indices) {
          // dim_indices is already sorted if self is coalesced and dim == 0
          if (self.is_coalesced() && dim == 0) {
            return std::make_tuple(dim_indices, at::arange(nnz, dim_indices.options()), nneg_index);
          }
          else {
            Tensor sorted_dim_indices, sorted_dim_indices_idx;
            std::tie(sorted_dim_indices, sorted_dim_indices_idx) = dim_indices.sort();
            return std::make_tuple(sorted_dim_indices, sorted_dim_indices_idx, nneg_index);
          }
        }
        // sort nneg_index to binary search into it
        else {
          Tensor sorted_nneg_index, sorted_nneg_index_idx;
          std::tie(sorted_nneg_index, sorted_nneg_index_idx) = nneg_index.sort();
          return std::make_tuple(sorted_nneg_index, sorted_nneg_index_idx, dim_indices);
        }
      }();

      const auto src_grain_size = at::internal::GRAIN_SIZE;
      const auto src_len = src.numel();
      const auto n_threads_src = std::max<int64_t>(
          // 1 <= n_threads_src <= std::min(ceil(src.numel() / src_grain_size), max_threads)
          1, std::min<int64_t>((src_len + src_grain_size - 1) / src_grain_size, at::get_num_threads())
      );
      const auto chunk_size_src = (src_len + n_threads_src - 1) / n_threads_src;

      const std::vector<int64_t> src_n_threads_shape = {
        n_threads_src, (src_len + n_threads_src - 1) / n_threads_src
      };

      // src_int_idx and sorted_int_idx store "i" and "j" indices indicating
      // intersections such that src_int_idx[i] == sorted_int_idx[j].
      // These intersections are found with binary search and in parallel.
      auto src_int_idx = at::empty(src_n_threads_shape, src.options());
      auto sorted_int_idx = at::empty_like(src_int_idx);
      // For each element "i" from src, int_counts define how many
      // elements there are in sorted, i.e. "j" indices, corresponding
      // to "i", i.e.:
      // |{j : src_int_idx[i] == sorted_int_idx[j]}| for each i in src_int_idx.
      auto int_counts = at::zeros_like(src_int_idx);

      // fill in src_int_idx, sorted_int_idx, int_counts
      {
        const auto sorted_len = sorted.numel();
        const auto* ptr_sorted = sorted.data_ptr<int64_t>();
        const auto* ptr_sorted_start = ptr_sorted;
        const auto* ptr_sorted_end = ptr_sorted + sorted_len;

        at::parallel_for(0, n_threads_src, 1, [&](int64_t tid, C10_UNUSED int64_t _) {
            const auto start = tid * chunk_size_src;
            const auto end = std::min(start + chunk_size_src, src_len);
            auto* ptr_tid_src_int_idx = src_int_idx.select(0, tid).data_ptr<int64_t>();
            auto* ptr_tid_sorted_int_idx = sorted_int_idx.select(0, tid).data_ptr<int64_t>();
            auto* ptr_tid_int_counts = int_counts.select(0, tid).data_ptr<int64_t>();
            const auto* ptr_src = src.data_ptr<int64_t>() + start;

            for (const auto i : c10::irange(start, end)) {
              const auto src_val = *ptr_src++;
              const auto src_val_lb = std::lower_bound(ptr_sorted_start, ptr_sorted_end, src_val);
              // We cannot just use *src_val_lb != src_val because when
              // src_val_lb == ptr_sorted_end, dereferencing past-the-end value
              // is not well-defined.
              if (src_val_lb == ptr_sorted_end || *src_val_lb != src_val) {
                ++ptr_tid_src_int_idx;
                ++ptr_tid_sorted_int_idx;
                ++ptr_tid_int_counts;
                continue;
              }
              const auto src_val_ub = std::upper_bound(ptr_sorted_start, ptr_sorted_end, src_val);

              const int64_t count = src_val_ub - src_val_lb;
              const int64_t j = src_val_lb - ptr_sorted_start;

              *ptr_tid_src_int_idx++ = i;
              *ptr_tid_sorted_int_idx++ = j;
              *ptr_tid_int_counts++ = count;
            }
        });
      }

      const auto compressed_int_counts = int_counts.sum(-1);
      const auto res_len = compressed_int_counts.sum().item<int64_t>();

      // Short-circuit if empty intersection
      if (!res_len) {
        auto empty_idx = at::empty({0}, src.options());
        return std::make_tuple(empty_idx, empty_idx);
      }

      // Now that we know "i", "j" and the counts, we "unflatten"
      // them into two arrays of intersection indices such that
      // selected_src = repeat_interleave(src_int_idx, int_counts),
      // and selected_sorted is obtained as follows:
      // offsets = int_counts.cumsum(0).sub_(int_counts)
      // for ii, (j, c) in enumerate(zip(sorted_int_idx, int_counts)):
      //     out_slice = slice(offsets[ii], offsets[ii] + c)
      //     src_slice = slice(j, j + c)
      //     selected_sorted[out_slice] = sorted_int_idx[src_slice]
      auto selected_sorted = at::empty({res_len}, sorted.options());
      auto selected_src = at::empty({res_len}, src.options());

      // fill in selected_sorted, selected_src
      {
        auto* ptr_selected_sorted = selected_sorted.data_ptr<int64_t>();
        auto* ptr_selected_src = selected_src.data_ptr<int64_t>();

        const auto thread_offsets = compressed_int_counts.cumsum(0).sub_(compressed_int_counts);
        const auto* ptr_sorted_idx = sorted_idx.data_ptr<int64_t>();
        at::parallel_for(0, n_threads_src, 1, [&](int64_t tid, C10_UNUSED int64_t _) {
            const auto start = tid * chunk_size_src;
            const auto end = std::min(start + chunk_size_src, src_len);
            const auto tid_offset = thread_offsets.data_ptr<int64_t>()[tid];
            const auto* ptr_tid_src_int_idx = src_int_idx.select(0, tid).data_ptr<int64_t>();
            const auto* ptr_tid_sorted_int_idx = sorted_int_idx.select(0, tid).data_ptr<int64_t>();
            const auto* ptr_tid_int_counts = int_counts.select(0, tid).data_ptr<int64_t>();
            auto* ptr_tid_selected_sorted = ptr_selected_sorted + tid_offset;
            auto* ptr_tid_selected_src = ptr_selected_src + tid_offset;

            for (C10_UNUSED const auto _ : c10::irange(start, end)) {
              const auto count = *ptr_tid_int_counts++;
              const auto i = *ptr_tid_src_int_idx++;
              const auto j = *ptr_tid_sorted_int_idx++;
              if (!count) continue;

              std::fill_n(ptr_tid_selected_src, count, i);
              std::copy_n(ptr_sorted_idx + j, count, ptr_tid_selected_sorted);

              ptr_tid_selected_sorted += count;
              ptr_tid_selected_src += count;
            }
        });
      }

      return search_in_dim_indices
        ? std::make_tuple(selected_sorted, selected_src)
        : std::make_tuple(selected_src, selected_sorted);
    };

    // Converts a 1d sorted idx to a compressed 1d compressed idx,
    // aka crow in the CSR format. Useful to get a count table in
    // a parallelized and no-sync manner.
    // TODO: this function is equivalent to _convert_indices_from_coo_to_csr.
    // The mentioned function is not public yet.
    const auto sorted_idx_to_cidx = [](
        const Tensor& idx,
        int64_t len,
        bool run_in_parallel = true) -> Tensor {
      auto cidx = at::empty({len + 1}, idx.options());

      const auto* ptr_idx = idx.data_ptr<int64_t>();
      auto* ptr_cidx = cidx.data_ptr<int64_t>();

      const auto idx_len = idx.numel();

      std::fill_n(ptr_cidx, ptr_idx[0] + 1, 0);
      std::fill_n(ptr_cidx + ptr_idx[idx_len - 1] + 1, len - ptr_idx[idx_len - 1], idx_len);

      const auto grain_size = run_in_parallel ? at::internal::GRAIN_SIZE : idx_len;
      at::parallel_for(0, idx_len, grain_size, [&](int64_t start, int64_t end) {
          auto* ptr_curr_cidx = ptr_cidx + ptr_idx[start] + 1;
          for (int64_t i = start; i < std::min(end, idx_len - 1); ++i) {
            const auto diff = ptr_idx[i + 1] - ptr_idx[i];
            std::fill_n(ptr_curr_cidx, diff, i + 1);
            ptr_curr_cidx += diff;
          }
      });

      return cidx;
    };

    // If nnz is (much) larger than size, then both indices[dim] and index get sorted
    // with a count sort (faster, and no huge nnz-sized chunk memory allocations).
    // The element-wise product between the count tables gives us all the intersections.
    const auto get_selected_indices_large_nnz_small_size = [&]() -> std::tuple<Tensor, Tensor> {
      const auto get_counts = [&sorted_idx_to_cidx](
          // Writes into counts (must be preallocated and zero)
          // and allows to use external buffers.
          Tensor& counts,
          const Tensor& t,
          int64_t bins,
          bool is_sorted = false,
          bool run_in_parallel = true) -> void {
        if (is_sorted) {
          const auto cidx = sorted_idx_to_cidx(t, bins, run_in_parallel);
          at::sub_out(counts, cidx.slice(0, 1, bins + 1), cidx.slice(0, 0, bins));
        }
        else {
          auto* ptr_counts = counts.data_ptr<int64_t>();
          const auto* ptr_vals = t.data_ptr<int64_t>();
          for (C10_UNUSED const auto _ : c10::irange(t.numel())) {
            ++ptr_counts[*ptr_vals++];
          }
        }
      };

      const auto counts_per_thread = [&get_counts, size](
          const Tensor& idx,
          bool is_sorted = false,
          int64_t grain_size = at::internal::GRAIN_SIZE
      ) -> Tensor {
        const auto idx_len = idx.numel();
        // 1 <= n_threads <= min(ceil(len / grain_size), max_threads)
        const auto n_threads = std::max<int64_t>(
            1, std::min<int64_t>((idx_len + grain_size - 1) / grain_size, at::get_num_threads())
        );
        const auto chunk_size = (idx_len + n_threads - 1) / n_threads;
        const auto run_in_parallel = (n_threads == 1);

        auto counts_per_thread = at::zeros({n_threads, size}, idx.options());
        at::parallel_for(0, n_threads, 1, [&](int64_t tid, C10_UNUSED int64_t _) {
          const auto start = tid * chunk_size;
          const auto end = std::min(start + chunk_size, idx_len);
          const auto tid_idx = idx.slice(0, start, end);
          auto tid_counts = counts_per_thread.select(0, tid);
          get_counts(tid_counts, tid_idx, /*bins=*/size,
              /*is_sorted=*/is_sorted, /*run_in_parallel=*/run_in_parallel);
        });

        return counts_per_thread;
      };

      auto dim_indices_counts_per_thread = counts_per_thread(
          dim_indices,
          /*is_sorted=*/self.is_coalesced() && dim == 0
          /*grain_size = at::internal::GRAIN_SIZE*/
      );
      auto dim_indices_offset_counts_per_thread = dim_indices_counts_per_thread.cumsum(0);

      auto index_counts_per_thread = counts_per_thread(
          nneg_index,
          /*is_sorted=*/false
          /*grain_size = at::internal::GRAIN_SIZE*/
      );
      auto index_offset_counts_per_thread = index_counts_per_thread.cumsum(0);

      const auto index_counts = index_offset_counts_per_thread.select(0, -1);
      const auto dim_indices_counts = dim_indices_offset_counts_per_thread.select(0, -1);
      const auto intersection_counts = index_counts.mul(dim_indices_counts);
      const auto res_len = intersection_counts.sum().item<int64_t>();
      // Short-circuit if empty intersection
      if (!res_len) {
        auto empty_idx = at::empty({0}, index.options());
        return std::make_tuple(empty_idx, empty_idx);
      }
      const auto intersection_offsets = intersection_counts.cumsum(0);

      const auto search_in_dim_indices = [&]() -> bool {
        const auto grain_size = at::internal::GRAIN_SIZE;
        const auto n_threads_index = std::max<int64_t>(
            1, std::min<int64_t>((index_len + grain_size - 1) / grain_size, at::get_num_threads())
        );
        const auto n_threads_dim_indices = std::max<int64_t>(
            1, std::min<int64_t>((nnz + grain_size - 1) / grain_size, at::get_num_threads())
        );

        const auto index_max_copy_work_per_thread =
          index_counts_per_thread.mul(dim_indices_counts).sum(-1).max().item<int64_t>();
        const auto dim_indices_max_copy_work_per_thread
          = dim_indices_counts_per_thread.mul(index_counts).sum(-1).max().item<int64_t>();

        const auto index_max_work_per_thread = index_max_copy_work_per_thread * index_len / n_threads_index;
        const auto dim_indices_max_work_per_thread = dim_indices_max_copy_work_per_thread * nnz / n_threads_dim_indices;
        return index_max_work_per_thread <= dim_indices_max_work_per_thread
          ? true
          : false;
      }();

      Tensor idx, idx_counts_per_thread, idx_offset_counts_per_thread;
      Tensor src, src_counts_per_thread, src_offset_counts_per_thread;
      std::tie(
          idx, idx_counts_per_thread, idx_offset_counts_per_thread,
          src, src_counts_per_thread, src_offset_counts_per_thread
      ) = [&]() {
        return search_in_dim_indices
          ? std::make_tuple(
              nneg_index, index_counts_per_thread, index_offset_counts_per_thread,
              dim_indices, dim_indices_counts_per_thread, dim_indices_offset_counts_per_thread
            )
          : std::make_tuple(
              dim_indices, dim_indices_counts_per_thread, dim_indices_counts_per_thread.cumsum(0),
              nneg_index, index_counts_per_thread, index_counts_per_thread.cumsum(0)
            );
      }();

      const auto idx_counts = idx_offset_counts_per_thread.select(0, -1);
      const auto src_counts = src_offset_counts_per_thread.select(0, -1);

      Tensor src_idx, src_idx_offsets;
      std::tie(src_idx, src_idx_offsets) = [&](
          int64_t grain_size = at::internal::GRAIN_SIZE
      ) -> std::tuple<Tensor, Tensor> {
        const auto src_intersection_counts = src_counts.mul(idx_counts > 0);
        const auto src_intersection_offsets = src_intersection_counts.cumsum(0);
        const auto src_idx_len = src_intersection_offsets.data_ptr<int64_t>()[size - 1];
        auto src_idx = at::empty({src_idx_len}, src.options());

        const auto* ptr_src = src.data_ptr<int64_t>();
        const auto* ptr_intersection_counts = intersection_counts.data_ptr<int64_t>();
        const auto* ptr_src_intersection_counts = src_intersection_counts.data_ptr<int64_t>();
        const auto* ptr_src_intersection_offsets = src_intersection_offsets.data_ptr<int64_t>();
        auto* ptr_src_idx = src_idx.data_ptr<int64_t>();

        const auto src_len = src.numel();
        const auto n_threads_src = std::max<int64_t>(
            1, std::min<int64_t>((src_len + grain_size - 1) / grain_size, at::get_num_threads())
        );
        const auto chunk_size = (src_len + n_threads_src - 1) / n_threads_src;
        at::parallel_for(0, n_threads_src, 1, [&](int64_t tid, C10_UNUSED int64_t _) {
            const auto start = tid * chunk_size;
            const auto end = std::min(start + chunk_size, src_len);
            auto* ptr_src_tid = ptr_src + start;
            const auto* ptr_src_counts_per_thread
              = src_counts_per_thread.select(0, tid).data_ptr<int64_t>();
            const auto* ptr_src_offset_counts_per_thread
              = src_offset_counts_per_thread.select(0, tid).data_ptr<int64_t>();
            auto tid_counts = at::zeros({size}, src.options());
            auto* ptr_tid_counts = tid_counts.data_ptr<int64_t>();

            for (const auto i : c10::irange(start, end)) {
              const auto idx_val = *ptr_src_tid++;
              // skip idx value if not in the intersection
              if (!ptr_intersection_counts[idx_val]) continue;
              const auto idx_val_offset
                = ptr_src_intersection_offsets[idx_val]
                - ptr_src_intersection_counts[idx_val];
              const auto idx_val_tid_offset
                = ptr_src_offset_counts_per_thread[idx_val]
                - ptr_src_counts_per_thread[idx_val];
              auto& idx_val_local_tid_count = ptr_tid_counts[idx_val];
              ptr_src_idx[idx_val_offset + idx_val_tid_offset + idx_val_local_tid_count] = i;
              ++idx_val_local_tid_count;
            }
        });

        const auto src_idx_offsets = src_intersection_offsets.sub_(src_intersection_counts);

        return std::make_tuple(src_idx, src_idx_offsets);
      }();

      Tensor idx_selected, src_selected;
      std::tie(idx_selected, src_selected) = [&](
          int64_t grain_size = at::internal::GRAIN_SIZE
      ) -> std::tuple<Tensor, Tensor> {
        const auto thread_offset = [&]() {
          // we do not need idx_counts_per_thread anymore,
          // so it is safe to do in-place intersection.
          auto counts_per_thread = idx_counts_per_thread.mul_(src_counts).sum(-1);
          return counts_per_thread.cumsum(0).sub_(counts_per_thread);
        }();
        const auto* ptr_thread_offset = thread_offset.data_ptr<int64_t>();

        auto idx_selected = at::empty({res_len}, idx.options());
        auto src_selected = at::empty({res_len}, src.options());

        const auto* ptr_idx = idx.data_ptr<int64_t>();
        const auto* ptr_src_counts = src_counts.data_ptr<int64_t>();
        const auto* ptr_intersection_counts = intersection_counts.data_ptr<int64_t>();
        const auto* ptr_src_idx = src_idx.data_ptr<int64_t>();
        const auto* ptr_src_idx_offsets = src_idx_offsets.data_ptr<int64_t>();
        auto* ptr_idx_selected = idx_selected.data_ptr<int64_t>();
        auto* ptr_src_selected = src_selected.data_ptr<int64_t>();

        const auto idx_len = idx.numel();
        const auto n_threads_idx = std::max<int64_t>(
            1, std::min<int64_t>((idx_len + grain_size - 1) / grain_size, at::get_num_threads())
        );
        const auto chunk_size = (idx_len + n_threads_idx - 1) / n_threads_idx;
        at::parallel_for(0, n_threads_idx, 1, [&](int64_t tid, C10_UNUSED int64_t _) {
            const auto start = tid * chunk_size;
            const auto end = std::min(start + chunk_size, idx_len);
            const auto tid_offset = ptr_thread_offset[tid];
            const auto* ptr_idx_tid = ptr_idx + start;
            auto* ptr_idx_selected_tid = ptr_idx_selected + tid_offset;
            auto* ptr_src_selected_tid = ptr_src_selected + tid_offset;

            for (const auto i : c10::irange(start, end)) {
              const auto idx_val = *ptr_idx_tid++;
              // skip if idx_val is not in the intersection
              if (!ptr_intersection_counts[idx_val]) continue;
              const auto count = ptr_src_counts[idx_val];
              const auto j = ptr_src_idx_offsets[idx_val];
              std::fill_n(ptr_idx_selected_tid, count, i);
              std::copy_n(ptr_src_idx + j, count, ptr_src_selected_tid);
              ptr_idx_selected_tid += count;
              ptr_src_selected_tid += count;
            }
        });

        return std::make_tuple(idx_selected, src_selected);
      }();

      return search_in_dim_indices
        ? std::make_tuple(src_selected, idx_selected)
        : std::make_tuple(idx_selected, src_selected);
    };

    const auto make_output = [&](
        const Tensor& selected_dim_indices,
        const Tensor& res_dim_indices) -> Tensor {
      auto res_indices = index_select(indices, 1, selected_dim_indices);
      res_indices[dim] = res_dim_indices;
      const auto res_values = index_select(values, 0, selected_dim_indices);

      return _sparse_coo_tensor_with_dims_and_tensors(
          sparse_dim, dense_dim, res_sizes, res_indices, res_values, self.options());
    };

    // Brute-force solution for small values of nnz and index_len
    const auto get_result_small_nnz_small_index = [&]()
      -> Tensor {
      const auto dim_indices_in_inner_loop = nnz >= index_len;
      Tensor outer, inner;
      std::tie(outer, inner) = [&]() -> std::tuple<Tensor, Tensor> {
        if (dim_indices_in_inner_loop) {
          return std::make_tuple(nneg_index, dim_indices);
        }
        else {
          return std::make_tuple(dim_indices, nneg_index);
        }
      }();

      const auto* ptr_outer = outer.data_ptr<int64_t>();
      const auto* ptr_inner = inner.data_ptr<int64_t>();
      // NOTE: if very critical, replace std::vector with
      // a data structure that operates on stack up to some limit.
      auto outer_selected_idx = std::vector<int64_t>();
      auto inner_selected_idx = std::vector<int64_t>();
      int64_t res_len = 0;
      for (const auto i : c10::irange(outer.numel())) {
        for (const auto j : c10::irange(inner.numel())) {
          if (ptr_outer[i] == ptr_inner[j]) {
            ++res_len;
            outer_selected_idx.push_back(i);
            inner_selected_idx.push_back(j);
          }
        }
      }

      const auto outer_selected_idx_tensor = at::from_blob(
          outer_selected_idx.data(), {res_len}, at::kLong
      );
      const auto inner_selected_idx_tensor = at::from_blob(
          inner_selected_idx.data(), {res_len}, at::kLong
      );

      return dim_indices_in_inner_loop
        ? make_output(inner_selected_idx_tensor, outer_selected_idx_tensor)
        : make_output(outer_selected_idx_tensor, inner_selected_idx_tensor);
    };

    constexpr int64_t BRUTE_FORCE_SIZE_LIMIT = 2 << 14; // 16384
    // NOTE: such a condition to avoid overflows in (nnz * index_len)
    if (nnz <= BRUTE_FORCE_SIZE_LIMIT && index_len <= BRUTE_FORCE_SIZE_LIMIT
        && (nnz * index_len) <= BRUTE_FORCE_SIZE_LIMIT) {
      return get_result_small_nnz_small_index();
    }
    else {
      Tensor selected_dim_indices;
      Tensor res_dim_indices;

      // A more precise decision could be of the form:
      // `nnz < C(nnz, size) * size`, but it requires heavy benchmarking.
      // We choose `nnz < size`, which measures theoretical complexity
      // and does not rely on runtime performance.
      // TODO: perform this analysis and find better C(nnz, size).
      if (nnz <= size) {
        std::tie(selected_dim_indices, res_dim_indices) = get_selected_indices_small_nnz_large_size();
      }
      else {
        std::tie(selected_dim_indices, res_dim_indices) = get_selected_indices_large_nnz_small_size();
      }

      return make_output(selected_dim_indices, res_dim_indices);
    }
  }
  // If indexing into dense dimensions
  else {
    // It is sufficient to just perform `index_select` on values
    // if `dim` refers to dense dimensions.
    const auto res_values = index_select(values, dim - sparse_dim + 1, index);

    return _sparse_coo_tensor_with_dims_and_tensors(
        sparse_dim, dense_dim, res_sizes, indices, res_values, self.options());
  }
}

Tensor slice(
    const Tensor& self,
    int64_t dim,
    c10::optional<int64_t> start,
    c10::optional<int64_t> end,
    int64_t step) {
  int64_t ndim = self.dim();
  if (ndim == 0) {
    TORCH_CHECK_INDEX(false, "slice() cannot be applied to a 0-dim tensor.");
  }
  dim = maybe_wrap_dim(dim, ndim);
  DimVector sizes(self.sizes().begin(), self.sizes().end());
  DimVector strides(self.strides().begin(), self.strides().end());
  // handle optional parameters
  int64_t start_val = start.has_value() ? start.value() : 0;
  int64_t end_val = end.has_value() ? end.value() : INT64_MAX;

  // TODO: support negative strides
  TORCH_CHECK(step > 0, "slice step must be positive");

  if (start_val < 0) {
    start_val += sizes[dim];
  }
  if (end_val < 0) {
    end_val += sizes[dim];
  }
  if (start_val < 0) {
    start_val = 0;
  } else if (start_val >= sizes[dim]) {
    start_val = sizes[dim];
  }
  if (end_val < start_val) {
    end_val = start_val;
  } else if (end_val >= sizes[dim]) {
    end_val = sizes[dim];
  }
  auto storage_offset = self.storage_offset() + start_val * strides[dim];
  auto len = end_val - start_val;
  sizes[dim] = (len + step - 1) / step; // round-up
  strides[dim] *= step;

  Tensor result;
  if (self.is_quantized()) {
    auto quantizer = create_subtensor_quantizer(self, false, start_val, end_val, dim, step);
    result = as_strided_qtensorimpl(self, sizes, strides, storage_offset, quantizer);
  } else {
    // NB: it is extremely important to perform a redispatch here for
    // the MPS backend; if you call directly to as_strided_tensorimpl,
    // the necessary metadata for MPS will not get setup and you will
    // get silently wrong results
    result = self.as_strided(sizes, strides, storage_offset);
  }
  namedinference::propagate_names(result, self);
  return result;
}

Tensor slice_backward(const Tensor& grad, IntArrayRef input_sizes, int64_t dim, int64_t start, int64_t end, int64_t step) {
  auto grad_input = at::zeros(input_sizes, grad.options());
  grad_input.slice(dim, start, end, step).copy_(grad);
  return grad_input;
}

std::vector<Tensor> split(const Tensor& self, int64_t split_size, int64_t dim) {
  const auto num_splits = get_num_splits(self, split_size, dim);
  std::vector<Tensor> splits(num_splits);
  int64_t last_split_size = split_size - (split_size * num_splits - self.size(dim));

  for (const auto i : c10::irange(num_splits)) {
    auto length = i < num_splits - 1 ? split_size : last_split_size;
    splits[i] = self.narrow(dim, i * split_size, length);
  }
  return splits;
}

std::vector<Tensor> split_symint(const Tensor& self, c10::SymIntArrayRef sizes, int64_t dim) {
  return at::split_with_sizes_symint(self, sizes, dim);
}

std::vector<Tensor> unsafe_split(const Tensor& self, int64_t split_size, int64_t dim) {
  auto result = at::native::split(self, split_size, dim);
  for (auto& t : result) {
    // TODO(Ailing): do we need to set version_counter here?
    if (!t.is_inference()) {
      t.unsafeGetTensorImpl()->set_version_counter(c10::VariableVersion(/*version=*/0));
    }
  }
  return result;
}

std::vector<Tensor> hsplit(const Tensor& self, int64_t split_size) {
  TORCH_CHECK(self.dim() >= 1, "torch.hsplit requires a tensor with at least 1 dimension, but got a tensor with ", self.dim(), " dimensions!")
  int64_t dim = (self.dim() == 1) ? 0 : 1;
  TORCH_CHECK(split_size != 0 && self.sizes()[dim] % split_size == 0,
    "torch.hsplit attempted to split along dimension ", dim,", but the size of the dimension ", self.sizes()[dim], " is not divisible by the split_size ", split_size, "!");
  return at::tensor_split(self, split_size, dim);
}

std::vector<Tensor> vsplit(const Tensor& self, int64_t split_size) {
  TORCH_CHECK(self.dim() >= 2, "torch.vsplit requires a tensor with at least 2 dimension, but got a tensor with ", self.dim(), " dimensions!")
  TORCH_CHECK(split_size != 0 && self.sizes()[0] % split_size == 0,
    "torch.vsplit attempted to split along dimension ", 0,", but the size of the dimension ", self.sizes()[0], " is not divisible by the split_size ", split_size, "!");
  return at::tensor_split(self, split_size, 0);
}

std::vector<Tensor> dsplit(const Tensor& self, int64_t split_size) {
  TORCH_CHECK(self.dim() >= 3, "torch.dsplit requires a tensor with at least 3 dimension, but got a tensor with ", self.dim(), " dimensions!")
  TORCH_CHECK(split_size != 0 && self.sizes()[2] % split_size == 0,
    "torch.dsplit attempted to split along dimension ", 2,", but the size of the dimension ", self.sizes()[2], " is not divisible by the split_size ", split_size, "!");
  return at::tensor_split(self, split_size, 2);
}

std::vector<Tensor> split_with_sizes(const Tensor& self, IntArrayRef split_sizes, int64_t dim) {
  TORCH_CHECK(self.dim() != 0, "split expects at least a 1-dimensional tensor");
  const int64_t dim_size = self.size(dim);
  const int64_t num_splits = split_sizes.size();
  int64_t start_idx = 0;

  std::vector<Tensor> splits;
  splits.reserve(num_splits);
  for (const auto i : c10::irange(num_splits)) {
    auto length = split_sizes[i];
    TORCH_CHECK(length >= 0,
             "split_with_sizes expects split_sizes have only non-negative ",
             "entries, but got split_sizes=", split_sizes);
    splits.push_back(at::native::slice(self, dim, start_idx, start_idx + length, 1));
    start_idx += length;
  }
  TORCH_CHECK(start_idx == dim_size,
           "split_with_sizes expects split_sizes to sum exactly to ", dim_size,
           " (input tensor's size at dimension ", dim, "), ", "but got split_sizes=", split_sizes);
  return splits;
}

std::vector<Tensor> unsafe_split_with_sizes(const Tensor& self, IntArrayRef split_sizes, int64_t dim) {
  auto result = at::native::split_with_sizes(self, split_sizes, dim);
  for (auto& t : result) {
    // TODO(Ailing): do we need to set version_counter here?
    if (!t.is_inference()) {
      t.unsafeGetTensorImpl()->set_version_counter(c10::VariableVersion(/*version=*/0));
    }
  }
  return result;
}

std::vector<Tensor> hsplit(const Tensor& self, IntArrayRef split_sizes) {
  TORCH_CHECK(self.dim() >= 1, "torch.hsplit requires a tensor with at least 1 dimension, but got a tensor with ", self.dim(), " dimensions!")
  return at::tensor_split(self, split_sizes, (self.dim() == 1) ? 0 : 1);
}

std::vector<Tensor> vsplit(const Tensor& self, IntArrayRef split_sizes) {
  TORCH_CHECK(self.dim() >= 2, "torch.vsplit requires a tensor with at least 2 dimension, but got a tensor with ", self.dim(), " dimensions!")
  return at::tensor_split(self, split_sizes, 0);
}

std::vector<Tensor> dsplit(const Tensor& self, IntArrayRef split_sizes) {
  TORCH_CHECK(self.dim() >= 3, "torch.dsplit requires a tensor with at least 3 dimension, but got a tensor with ", self.dim(), " dimensions!")
  return at::tensor_split(self, split_sizes, 2);
}

// Precondition: tensors is non-empty
static inline std::vector<Tensor> get_stack_inputs(TensorList tensors, int64_t dim) {
  std::vector<Tensor> inputs(tensors.size());
  at::IntArrayRef entry_shape = tensors[0].sizes();
  inputs[0] = tensors[0].unsqueeze(dim);
  for (const auto i : c10::irange(1, tensors.size())) {
    TORCH_CHECK(tensors[i].sizes() == entry_shape,
      "stack expects each tensor to be equal size, but got ", entry_shape,
      " at entry 0 and ", tensors[i].sizes(), " at entry ", i);
    inputs[i] = tensors[i].unsqueeze(dim);
  }
  return inputs;
}

bool inline maybe_native_stack(Tensor& result, TensorList tensors, int64_t dim) {
  dim = maybe_wrap_dim(dim, tensors[0].dim() + 1);
  if (detail::CanUseNativeSerialStack<TensorList, /*skip_overlap_check*/ false>::call(result, tensors, dim)) {
    // compute the size of the result
    auto result_sizes = tensors[0].sizes().vec();
    result_sizes.insert(result_sizes.begin() + dim, tensors.size());

    // skip resizing if size of result is same as expected
    // raise a warning while resizing if output has one or more elements
    // at::native::resize_output(result, result_sizes);
    // TODO: restore the above, see https://github.com/pytorch/pytorch/issues/64709

    if (result.sizes() != result_sizes) {
      result.resize_(result_sizes);
    }

    stack_serial_stub(kCPU, result, tensors, dim);
    return true;
  }
  return false;
}

Tensor _stack(TensorList tensors, int64_t dim) {
  ScalarType high_type = result_type(tensors);
  Tensor result = at::empty({0}, tensors[0].options().dtype(high_type));
  return at::native::_stack_out(get_stack_inputs(tensors, dim), dim, result);
}

Tensor _stack_cpu(TensorList tensors, int64_t dim) {
  ScalarType high_type = result_type(tensors);
  Tensor result = at::empty({0}, tensors[0].options().dtype(high_type));
  return at::native::_stack_out_cpu(tensors, dim, result);
}

void check_stack_inputs(TensorList tensors, int64_t dim) {
  at::IntArrayRef entry_shape = tensors[0].sizes();
  for (const auto i : c10::irange(1, tensors.size())) {
    TORCH_CHECK(tensors[i].sizes() == entry_shape,
      "stack expects each tensor to be equal size, but got ", entry_shape,
      " at entry 0 and ", tensors[i].sizes(), " at entry ", i);
  }
}

// TODO(msubkhankulov): refactor to use _stack
Tensor stack(TensorList tensors, int64_t dim) {
  TORCH_CHECK(tensors.size() > 0,
           "stack expects a non-empty TensorList");
  auto wrapped_dim = maybe_wrap_dim(dim, tensors[0].ndimension()+1);
  if (wrapped_dim < tensors[0].ndimension() && !tensors[0].is_sparse()) {
    check_stack_inputs(tensors, wrapped_dim);
    auto result_sizes = tensors[0].sizes().vec();
    result_sizes.insert(result_sizes.begin() + wrapped_dim, tensors.size());
    auto out = at::cat(tensors, wrapped_dim);
    return out.view(result_sizes); // one can always split a dimension with view
  } else { //dim = tensors[0].ndimension() cannot be efficiently handled by view
    return at::cat(get_stack_inputs(tensors, dim), dim);
  }
}

// CPU specific implementation
Tensor& _stack_out_cpu(TensorList tensors, int64_t dim, Tensor& result) {
  if (maybe_native_stack(result, tensors, dim)) {
    return result;
  } else {
    return at::cat_out(result, get_stack_inputs(tensors, dim), dim);
  }
}

// default backend
Tensor& _stack_out(TensorList tensors, int64_t dim, Tensor& result) {
  return at::cat_out(result, tensors, dim);
}

// TODO(msubkhankulov): refactor to use _stack_out
Tensor& stack_out(TensorList tensors, int64_t dim, Tensor& result) {
  TORCH_CHECK(tensors.size() > 0,
           "stack expects a non-empty TensorList");
  auto wrapped_dim = maybe_wrap_dim(dim, tensors[0].ndimension()+1);
  if (wrapped_dim < tensors[0].ndimension() && !tensors[0].is_sparse()) {
    check_stack_inputs(tensors, wrapped_dim);
    auto result_sizes = tensors[0].sizes().vec();
    result_sizes.insert(result_sizes.begin() + wrapped_dim, tensors.size());
    at::native::resize_output(result, result_sizes);
    auto cat_sizes = tensors[0].sizes().vec();
    cat_sizes[wrapped_dim] *= tensors.size();
    auto strides = at::detail::computeStride(result.sizes(), result.strides(), cat_sizes);
    if (strides.has_value()) {
      //can take fast cat path
      auto result_view = result.view(cat_sizes);
      at::cat_out(result_view, tensors, wrapped_dim);
      return result;
    }
  }
  return at::cat_out(result, get_stack_inputs(tensors, dim), dim);

}

Tensor hstack(TensorList tensors) {
  TORCH_CHECK(tensors.size() > 0,
           "hstack expects a non-empty TensorList");
  auto rep = at::atleast_1d(tensors);
  if (rep[0].dim() == 1) {
    return at::cat(rep, 0);
  }
  return at::cat(rep, 1);
}

Tensor& hstack_out(TensorList tensors, Tensor& result) {
  TORCH_CHECK(tensors.size() > 0,
           "hstack expects a non-empty TensorList");
  auto rep = at::atleast_1d(tensors);
  if (rep[0].dim() == 1) {
    return at::cat_out(result, rep, 0);
  }
  return at::cat_out(result, rep, 1);
}

Tensor vstack(TensorList tensors) {
  TORCH_CHECK(tensors.size() > 0,
           "vstack expects a non-empty TensorList");
  auto rep = at::atleast_2d(tensors);
  return at::cat(rep, 0);
}

Tensor& vstack_out(TensorList tensors, Tensor& result) {
  TORCH_CHECK(tensors.size() > 0,
           "vstack expects a non-empty TensorList");
  auto rep = at::atleast_2d(tensors);
  return at::cat_out(result, rep, 0);
}

Tensor dstack(TensorList tensors) {
  TORCH_CHECK(tensors.size() > 0,
           "dstack expects a non-empty TensorList");
  auto rep = at::atleast_3d(tensors);
  return at::cat(rep, 2);
}
Tensor& dstack_out(TensorList tensors, Tensor& result) {
  TORCH_CHECK(tensors.size() > 0,
           "dstack expects a non-empty TensorList");
  auto rep = at::atleast_3d(tensors);
  return at::cat_out(result, rep, 2);
}

static inline Tensor & sparse_transpose_(Tensor & self, int64_t dim0, int64_t dim1) {
  int64_t nsparse_dim = self.sparse_dim();
  TORCH_CHECK(dim0 < nsparse_dim && dim1 < nsparse_dim,
           "sparse transpose: transposed dimensions must be sparse ",
           "Got sparse_dim: ", nsparse_dim, ", d0: ", dim0, ", d1: ", dim1);

  if (self._indices().numel() == 0 && self._values().numel() == 0) {
    auto sizes = self.sizes().vec();
    std::swap(sizes[dim0], sizes[dim1]);

    at::sparse::get_sparse_impl(self)->raw_resize_(self.sparse_dim(), self.dense_dim(), sizes);
  } else {
    auto indices = self._indices();
    auto row0 = indices.select(0, dim0);
    auto row1 = indices.select(0, dim1);

    // swap row0 and row1
    auto tmp = at::zeros_like(row0, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    tmp.copy_(row0);
    row0.copy_(row1);
    row1.copy_(tmp);

    self._coalesced_(false);

    auto sizes = self.sizes().vec();
    std::swap(sizes[dim0], sizes[dim1]);

    at::sparse::get_sparse_impl(self)->raw_resize_(self._indices().size(0), self._values().dim() - 1, sizes);
  }
  return self;
}

// torch.row_stack, alias for torch.vstack
Tensor& row_stack_out(TensorList tensors, Tensor& result) {
  return at::vstack_out(result, tensors);
}

Tensor row_stack(TensorList tensors) {
  return at::vstack(tensors);
}

static std::vector<Tensor> reshape_input_for_column_stack(TensorList tensors) {
  std::vector<Tensor> result(tensors.size());
  auto transform_lambda = [](const Tensor& input) -> Tensor {
    // reshape 0D or 1D tensor t into (t.numel(), 1)
    if (input.dim() <= 1) {
      return input.reshape({input.numel(), 1});
    }
    return input;
  };
  std::transform(tensors.cbegin(),
                 tensors.cend(),
                 result.begin(),
                 transform_lambda);
  return result;
}

Tensor& column_stack_out(TensorList tensors, Tensor& result) {
  TORCH_CHECK(tensors.size() > 0,
              "column_stack expects a non-empty TensorList");

  auto reshaped_tensors = reshape_input_for_column_stack(tensors);
  return at::hstack_out(result, reshaped_tensors);
}

Tensor column_stack(TensorList tensors) {
  TORCH_CHECK(tensors.size() > 0,
              "column_stack expects a non-empty TensorList");

  auto reshaped_tensors = reshape_input_for_column_stack(tensors);
  return at::hstack(reshaped_tensors);
}

static Tensor& propagate_transposed_names(
    Tensor& result,
    const Tensor& other,
    int64_t dim0,
    int64_t dim1) {
  if (other.has_names()) {
    auto names = other.names().vec();
    std::swap(names[dim0], names[dim1]);
    namedinference::propagate_names_if_nonempty(result, names);
  }
  return result;
}

Tensor transpose(const Tensor& self, Dimname dim0, Dimname dim1) {
  return at::transpose(
      self, dimname_to_position(self, dim0), dimname_to_position(self, dim1));
}


Tensor & transpose_(Tensor & self, int64_t dim0, int64_t dim1) {
  TORCH_CHECK(
      !(self.layout() == kSparseCsr || self.layout() == kSparseCsc ||
        self.layout() == kSparseBsr || self.layout() == kSparseBsc),
      "torch.transpose_: in-place transposition is not supported for ",
      self.layout(),
      " layout");

  auto ndims = self.dim();
  dim0 = maybe_wrap_dim(dim0, ndims);
  dim1 = maybe_wrap_dim(dim1, ndims);
  if (dim0 == dim1) {
    return self;
  }

  // Sparse COO is an exceptional sparse format as it allows transpose
  // to be a view operation which is a convinient property for
  // in-place operations. For other sparse formats, the in-place
  // transpose would not be possible without shuffling the specified
  // values. So we don't support this as it would defeat the purpose
  // of in-place opeations of being memory-efficient.
  if (self.is_sparse()) {
    return sparse_transpose_(self, dim0, dim1);
  }

  if (self.is_mkldnn()) {
    return at::_mkldnn_transpose_(self, dim0, dim1);
  }

  DimVector sizes(self.sizes().begin(), self.sizes().end());
  DimVector strides(self.strides().begin(), self.strides().end());
  std::swap(strides[dim0], strides[dim1]);
  std::swap(sizes[dim0], sizes[dim1]);
  self.as_strided_(sizes, strides);
  return self;
}

namespace {
// Transpose implementation for sparse compressed layouts
// NB: We assume that dim1,dim0 have already been wrapped
static inline Tensor sparse_compressed_transpose(
    const Tensor& self,
    int64_t dim0,
    int64_t dim1) {
  auto compressed_inds = AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(
      self.layout(),
      "compressed_inds",
      [&self]() { return self.crow_indices(); },
      [&self]() { return self.ccol_indices(); });

  auto plain_inds = AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(
      self.layout(),
      "plain_inds",
      [&self]() { return self.col_indices(); },
      [&self]() { return self.row_indices(); });

  const auto n_batch_dim = compressed_inds.dim() - 1;
  const auto n_dense_dim = self.dim() - n_batch_dim - 2;

  // In theory it works, but missing to_dense coverage to test
  TORCH_CHECK(
      n_dense_dim == 0,
      "transpose(): hybrid sparse compressed tensors with dense dimensions are not supported");

  // Classify transpose "type"
  enum class TransposeDim : uint8_t { Batch, Sparse, Dense };
  auto classify_dim = [&n_batch_dim](const int64_t dim) {
    if (dim < n_batch_dim) {
      return TransposeDim::Batch;
    } else if (dim > n_batch_dim + 1) {
      return TransposeDim::Dense;
    } else {
      return TransposeDim::Sparse;
    }
  };

  const auto transpose_type = classify_dim(dim0);
  {
    auto dim_type_name = [](const TransposeDim dim) {
      switch (dim) {
        case TransposeDim::Batch:
          return "Batch";
        case TransposeDim::Dense:
          return "Dense";
        case TransposeDim::Sparse:
          return "Sparse";
        default:
          TORCH_INTERNAL_ASSERT(
              false,
              "Impossible TransposeDim value: ",
              static_cast<std::underlying_type_t<TransposeDim>>(dim));
      }
    };
    const auto dim1_type = classify_dim(dim1);
    TORCH_CHECK(
        dim1_type == transpose_type,
        "transpose(): can only transpose dimensions of the same type (Batch, Sparse, Dense), got ",
        dim0,
        "(",
        dim_type_name(transpose_type),
        ")",
        " and ",
        dim1,
        "(",
        dim_type_name(dim1_type),
        ")");
  }

  // We have validated everything, early exit for equal dims (no effect)
  if (dim0 == dim1) {
    return self.clone();
  }

  auto result_sizes = DimVector(self.sizes());
  std::swap(result_sizes[dim0], result_sizes[dim1]);
  Tensor result_vals;
  auto result_layout = self.layout();

  if (transpose_type == TransposeDim::Batch) {
    compressed_inds = compressed_inds.transpose(dim0, dim1).contiguous();
    plain_inds = plain_inds.transpose(dim0, dim1).contiguous();
    result_vals = self.values().transpose(dim0, dim1).contiguous();

  } else if (transpose_type == TransposeDim::Dense) {
    // NB: This code should work, but is untestable due to lack of support for
    // dense dimensions in to_dense. The Debug assert is present to emphasize
    // the fact that the block should not be possible to hit this code block
    TORCH_INTERNAL_ASSERT(
        false, "transpose(): Shouldn't have reached this point");
    result_vals = AT_DISPATCH_PLAIN_SPARSE_COMPRESSED_LAYOUTS(
        self.layout(),
        "sparse_transpose",
        // un-blocked: 2 sparse dims map to single nnz dim, so dense dim0/1 are
        // one position left
        [&]() { return self.values().transpose(dim0 - 1, dim1 - 1); },
        // blocked: 2 sparse dims map to 3 (nnz, ) + blocksize dims, so dense
        // dim0/1 are one position right
        [&]() { return self.values().transpose(dim0 + 1, dim1 + 1); });
  } else /*if (transpose_type == TransposeDim::Sparse) */ {
    // Flip the layout
    result_layout = sparse_csr::flip_compressed_layout(self.layout());
    result_vals = AT_DISPATCH_PLAIN_SPARSE_COMPRESSED_LAYOUTS(
        self.layout(),
        "sparse_transpose",
        // un-blocked: no change to values, layout is flipped.
        [&]() { return self.values(); },
        // blocked: the blocks are nested under the sparse dims so they must be
        // transposed as well.
        [&]() {
          return self.values().transpose(-2 - n_dense_dim, -1 - n_dense_dim);
        });
  }
  return at::native::_sparse_compressed_tensor_unsafe(
      compressed_inds,
      plain_inds,
      result_vals,
      result_sizes,
      self.scalar_type(),
      result_layout,
      self.device());
}
} // namespace

Tensor transpose(const Tensor & self, int64_t dim0, int64_t dim1) {
  auto ndims = self.dim();
  dim0 = maybe_wrap_dim(dim0, ndims);
  dim1 = maybe_wrap_dim(dim1, ndims);

  if (self.is_sparse()) {
    if (dim0 == dim1) {
      return self.clone();
    }
    Tensor self_clone = self.clone();
    return sparse_transpose_(self_clone, dim0, dim1);
  }
  if (self.layout() == kSparseBsr || self.layout() == kSparseCsr ||
      self.layout() == kSparseBsc || self.layout() == kSparseCsc) {
    return sparse_compressed_transpose(self, dim0, dim1);
  }

  if (self.is_mkldnn()) {
    return at::_mkldnn_transpose(self, dim0, dim1);
  }

  // Transpose of a tensor is a view operation.
  if (dim0 == dim1) {
    return self.alias();
  }

  SymDimVector sizes(self.sym_sizes().begin(), self.sym_sizes().end());
  std::swap(sizes[dim0], sizes[dim1]);
  SymDimVector strides(self.sym_strides().begin(), self.sym_strides().end());
  std::swap(strides[dim0], strides[dim1]);
  auto result = self.as_strided_symint(sizes, strides);
  propagate_transposed_names(result, self, dim0, dim1);
  return result;
}

static void check_t(const Tensor& self, const char *fn) {
  if (self.is_sparse()) {
    int64_t sparse_dim = self.sparse_dim();
    int64_t dense_dim = self.dense_dim();
    TORCH_CHECK(sparse_dim <= 2 && dense_dim == 0,
             fn, " expects a tensor with <= 2 sparse and 0 dense dimensions, but got ",
             sparse_dim, " sparse and ", dense_dim, " dense dimensions");
  } else {
    TORCH_CHECK(self.dim() <= 2,
             fn, " expects a tensor with <= 2 dimensions, but self is ", self.dim(), "D");
  }
}

Tensor t(const Tensor & self) {
  check_t(self, "t()");
  return self.transpose(0, self.dim() < 2 ? 0 : 1);
}

Tensor & t_(Tensor & self) {
  check_t(self, "t_()");
  return self.transpose_(0, self.dim() < 2 ? 0 : 1);
}

std::tuple<SymDimVector, SymDimVector>
inferSqueezeGeometry(const Tensor &tensor) {
  SymDimVector sizes;
  SymDimVector strides;

  for(const auto d : c10::irange(tensor.dim())) {
    if(tensor.sym_sizes()[d] != 1) {
      sizes.push_back(tensor.sym_sizes()[d]);
      strides.push_back(tensor.sym_strides()[d]);
    }
  }

  return std::make_tuple(std::move(sizes), std::move(strides));
}

std::tuple<SymDimVector, SymDimVector>
inferSqueezeGeometry(const Tensor& tensor, int64_t dim) {
  SymDimVector sizes;
  SymDimVector strides;

  for(const auto d : c10::irange(tensor.dim())) {
    if(d != dim || tensor.sym_sizes()[dim] != 1) {
      sizes.push_back(tensor.sym_sizes()[d]);
      strides.push_back(tensor.sym_strides()[d]);
    }
  }
  return std::make_tuple(std::move(sizes), std::move(strides));
}

namespace {
// Named type instead of a pair/tuple so that we can be sure to
// construct the vectors in place and get NRVO.
struct InferUnsqueezeGeometryResult {
  DimVector sizes;
  DimVector strides;
  InferUnsqueezeGeometryResult(IntArrayRef tensor_sizes, IntArrayRef tensor_strides)
      : sizes(tensor_sizes.begin(), tensor_sizes.end())
      , strides(tensor_strides.begin(), tensor_strides.end()) {}
};
}
InferUnsqueezeGeometryResult
inferUnsqueezeGeometry(const Tensor& tensor, int64_t dim) {
  InferUnsqueezeGeometryResult result(tensor.sizes(), tensor.strides());
  int64_t new_stride = dim >= tensor.dim() ? 1 : result.sizes[dim] * result.strides[dim];
  result.sizes.insert(result.sizes.begin() + dim, 1);
  result.strides.insert(result.strides.begin() + dim, new_stride);

  return result;
}

// dim is present if squeezing a single dimension and absent if squeezing all dimensions
Tensor squeeze_qtensor(const Tensor& self, c10::optional<int64_t> dim) {
  auto quantizer = get_qtensorimpl(self)->quantizer();
  SymDimVector sizes;
  SymDimVector strides;
  std::tie(sizes, strides) = dim.has_value() ? inferSqueezeGeometry(self, dim.value()) : inferSqueezeGeometry(self);
  if (quantizer->qscheme() == QScheme::PER_CHANNEL_AFFINE) {
    const auto* per_channel_quantizer = static_cast<at::PerChannelAffineQuantizer*>(quantizer.get());
    auto axis = per_channel_quantizer->axis();
    int64_t shift = 0;
    integer_range<int64_t> dims = dim.has_value() ? integer_range<int64_t>{dim.value(), dim.value() + 1} : c10::irange(self.dim());
    for (const auto d : dims) {
      if (self.sizes()[d] == 1) {
        TORCH_CHECK(axis != d, "Squeeze is only possible on non-axis dimension for Per-Channel Quantized Tensors.");
        if (d < axis) {
          ++shift;
        }
      }
    }
    axis -= shift;
    quantizer = make_per_channel_affine_quantizer(per_channel_quantizer->scales(),
                                                  per_channel_quantizer->zero_points(),
                                                  axis,
                                                  quantizer->scalar_type());
  }
  // TODO: quantized Tensor support for SymInt needs to be added but basic building blocs
  // are missing for now.
  auto result = make_qtensor(self, c10::asIntArrayRefSlow(sizes), c10::asIntArrayRefSlow(strides), quantizer);
  if (dim.has_value()) {
    namedinference::propagate_names_except(result, self, {dim.value()});
  } else {
    auto maybe_outnames = namedinference::compute_squeeze_outnames(self);
    namedinference::propagate_names_if_nonempty(result, maybe_outnames);
  }

  return result;
}

Tensor squeeze(const Tensor& self) {
  auto g = inferSqueezeGeometry(self);
  at::Tensor result = self.as_strided_symint(std::get<0>(g), std::get<1>(g));
  auto maybe_outnames = namedinference::compute_squeeze_outnames(self);
  namedinference::propagate_names_if_nonempty(result, maybe_outnames);
  return result;
}

Tensor squeeze_quantized(const Tensor& self) {
  at::Tensor result = squeeze_qtensor(self, c10::nullopt);
  auto maybe_outnames = namedinference::compute_squeeze_outnames(self);
  namedinference::propagate_names_if_nonempty(result, maybe_outnames);
  return result;
}

Tensor squeeze(const Tensor& self, int64_t dim) {
  int64_t dims = self.dim();
  dim = maybe_wrap_dim(dim, dims);
  if (dims == 0 || self.sym_sizes()[dim] != 1) {
    return self.as_strided_symint(self.sym_sizes(), self.sym_strides());
  }
  auto g = inferSqueezeGeometry(self, dim);
  auto result = self.as_strided_symint(std::get<0>(g), std::get<1>(g));
  namedinference::propagate_names_except(result, self, {dim});
  return result;
}

Tensor squeeze_quantized(const Tensor& self, int64_t dim) {
  int64_t dims = self.dim();
  dim = maybe_wrap_dim(dim, dims);
  return squeeze_qtensor(self, dim);
}

Tensor & squeeze_(Tensor& self) {
  auto g = inferSqueezeGeometry(self);
  self.as_strided__symint(std::get<0>(g), std::get<1>(g));
  return self;
}

Tensor & squeeze_(Tensor& self, int64_t dim) {
  int64_t dims = self.dim();
  dim = maybe_wrap_dim(dim, self.dim());

  if (dims == 0 || self.sym_sizes()[dim] != 1) {
    self.as_strided__symint(self.sym_sizes(), self.sym_strides());
    return self;
  }
  auto g = inferSqueezeGeometry(self, dim);
  self.as_strided__symint(std::get<0>(g), std::get<1>(g));
  return self;
}

// NOTE [ Unsafe View ]
// _unsafe_view() differs from view() in that the returned tensor isn't treated
// as a view for the purposes of automatic differentiation. (It's not listed in
// VIEW_FUNCTIONS in gen_inplace_or_view_type.py).  It's only safe to use if the `self` tensor
// is temporary. For example, the viewed tensor here (a + b) is discarded immediately
// after viewing:
//
//  res = at::_unsafe_view(a + b, size);
//
// This is a hack because in-place operations on tensors treated like views
// can be much more expensive than the same operations on non-view tensors.

inline Tensor view_impl(const Tensor& self, IntArrayRef size) {

  at::DimVector inferred_size = at::infer_size_dv(size, self.numel());
  auto stride = at::detail::computeStride(self.sizes(),
                                          self.strides(),
                                          inferred_size);
  TORCH_CHECK(stride.has_value(), "view size is "
    "not compatible with input tensor's size and stride (at least one dimension"
    " spans across two contiguous subspaces). Use .reshape(...) instead.");
  return alias_with_sizes_and_strides(self, inferred_size, *stride);

}

Tensor _unsafe_view(const Tensor& self, IntArrayRef size) {
  return view_impl(self, size);
}

Tensor unsqueeze(const Tensor& self, int64_t dim) {
  dim = maybe_wrap_dim(dim, self.dim() + 1);
  auto g = inferUnsqueezeGeometry(self, dim);
  return self.as_strided(g.sizes, g.strides);
}

Tensor unsqueeze_sparse(Tensor const &self, int64_t dim) {
  dim = maybe_wrap_dim(dim, self.dim() + 1);
  int64_t sparse_dim = self.sparse_dim();
  int64_t dense_dim = self.dense_dim();
  auto indices = self._indices();
  auto sizes = self.sizes().vec();
  sizes.insert(sizes.begin() + dim, 1);
  if (dim <= sparse_dim) {
    auto new_indices = at::cat(
        {indices.narrow(0, 0, dim),
         at::zeros(
             {1, indices.size(1)},
             kLong,
             indices.options().layout_opt(),
             indices.options().device_opt(),
             indices.options().pinned_memory_opt()),
         indices.narrow(0, dim, indices.size(0) - dim)});
    return _sparse_coo_tensor_with_dims_and_tensors(
        sparse_dim + 1, dense_dim, sizes, new_indices, self._values(), self.options());
  } else {
    return _sparse_coo_tensor_with_dims_and_tensors(
        sparse_dim, dense_dim + 1, sizes, indices, self._values().unsqueeze(dim - sparse_dim + 1), self.options());
  }
}

Tensor unsqueeze_quantized(const Tensor& self, int64_t dim) {
  dim = maybe_wrap_dim(dim, self.dim() + 1);
  auto g = inferUnsqueezeGeometry(self, dim);
  auto quantizer = get_qtensorimpl(self)->quantizer();
  if (quantizer->qscheme() == QScheme::PER_CHANNEL_AFFINE) {
    const auto* per_channel_quantizer = static_cast<at::PerChannelAffineQuantizer*>(quantizer.get());
    auto axis = per_channel_quantizer->axis();
    if (axis >= dim) {
      axis += 1;
    }
    quantizer = make_per_channel_affine_quantizer(per_channel_quantizer->scales(),
                                                  per_channel_quantizer->zero_points(),
                                                  axis,
                                                  quantizer->scalar_type());
  }
  return make_qtensor(self, g.sizes, g.strides, quantizer);
}

Tensor & unsqueeze_(Tensor& self, int64_t dim) {
  dim = maybe_wrap_dim(dim, self.dim() + 1);

  auto g = inferUnsqueezeGeometry(self, dim);
  self.as_strided_(g.sizes, g.strides);
  return self;
}

Tensor flatten(const Tensor& self, int64_t start_dim, int64_t end_dim) {
  start_dim = maybe_wrap_dim(start_dim, self.dim());
  end_dim = maybe_wrap_dim(end_dim, self.dim());
  TORCH_CHECK(start_dim <= end_dim, "flatten() has invalid args: start_dim cannot come after end_dim");

  if (self.dim() == 0) {
    return self.reshape({1});
  }
  if (start_dim == end_dim) {
    return self;
  }

  // We don't want to infer_size on the entire shape, because that can give us an extra degree
  // of freedom we don't want; for example, consider shape [0, 1, 3, 0], with start_dim=1, end_dim=2.
  // It's clear we want result shape [0, 3, 0] but passing [0, -1, 0] to infer_size means the -1
  // can take on any value and satisfy the constraints.
  auto slice_numel = c10::multiply_integers(self.sym_sizes().slice(start_dim, end_dim - start_dim + 1));
  std::vector<c10::SymInt> shape;
  shape.reserve(self.dim() - end_dim + start_dim);
  for (const auto i : c10::irange(start_dim)) {
    shape.push_back(self.sym_sizes()[i]);
  }
  shape.push_back(slice_numel);
  for (const auto i : c10::irange(end_dim + 1, self.dim())) {
    shape.push_back(self.sym_sizes()[i]);
  }

  return native::reshape_symint(self, shape);
}

Tensor flatten(const Tensor& self, int64_t start_dim, int64_t end_dim, Dimname out_dim) {
  auto outnames = self.names().vec();
  outnames.erase(outnames.begin() + start_dim, outnames.begin() + end_dim + 1);
  outnames.insert(outnames.begin() + start_dim, out_dim);

  Tensor result;
  {
    NoNamesGuard guard;
    result = native::flatten(self, start_dim, end_dim);
  }
  internal_set_names_inplace(result, outnames);
  return result;
}

Tensor flatten(const Tensor& self, Dimname start_dim, Dimname end_dim, Dimname out_dim) {
  auto start_pos = dimname_to_position(self, start_dim);
  auto end_pos  = dimname_to_position(self, end_dim);
  return native::flatten(self, start_pos, end_pos, out_dim);
}

Tensor flatten(const Tensor& self, DimnameList dims, Dimname out_dim) {
  auto positions = dimnames_to_positions(self, dims);
  TORCH_CHECK(positions.size() > 0,
      "flatten(tensor, dims, out_dim): dims cannot be empty");
  for (const auto i : c10::irange(positions.size() - 1)) {
    if (positions[i] + 1 == positions[i + 1]) continue;
    TORCH_CHECK(positions[i] + 1 == positions[i + 1],
        "flatten(tensor, dims, out_dim): dims ", dims, " must be consecutive ",
        "in Tensor", self.names());
  }
  return native::flatten(self, *dims.begin(), *(dims.end() - 1), out_dim);
}

Tensor ravel(const Tensor& self) {
  return self.contiguous().view(-1);
}

static inline void handle_unflatten_exception(const std::runtime_error &e,
                                              const Tensor &self,
                                              int64_t dim,
                                              IntArrayRef sizes,
                                              c10::optional <DimnameList> names) {
  if (!strstr(e.what(), "is invalid for input of size")) {
    TORCH_CHECK(false, "unflatten got an unexpected error:\n", e.what());
  }

  if (self.has_names()) {
    TORCH_CHECK(false,
                "unflatten: Provided sizes ", sizes, " don't multiply up to the size of dim ",
                dim, " (", self.names()[dim], ": ", self.size(dim), ") in Tensor", self.names());

  } else {
    TORCH_CHECK(false,
                "unflatten: Provided sizes ", sizes, " don't multiply up to the size of dim ",
                dim, " (", self.size(dim), ") in the input tensor");
  }
}

Tensor unflatten_impl(const Tensor& self, int64_t dim, IntArrayRef sizes, c10::optional<DimnameList> names) {
  dim = maybe_wrap_dim(dim, self.dim());

  TORCH_CHECK(sizes.size() > 0, "unflatten: sizes must be non-empty");
  TORCH_INTERNAL_ASSERT(!names || names->size() == sizes.size());
  if (self.has_names()) {
    TORCH_CHECK(names, "unflatten: input is a named tensor but no names were given for unflattened sizes");
  }

  DimVector inferred_size;
  try {
    inferred_size = at::infer_size_dv(sizes, self.size(dim));
  } catch (const std::runtime_error& e) {
    // at::infer_size would throw std::runtime_error for invalid size,
    // catch the runtime_error and display the error message in a more user-friendly way
    // for both tensors and named tensors
    handle_unflatten_exception(e, self, dim, sizes, names);
  }

  DimVector shape(self.sizes().begin(), self.sizes().end());
  shape.erase(shape.begin() + dim);
  shape.insert(shape.begin() + dim, inferred_size.begin(), inferred_size.end());

  Tensor result;
  {
    NoNamesGuard guard;
    result = self.view(shape);
  }

  if (names) {
    auto outnames = self.names().vec();
    outnames.erase(outnames.begin() + dim);
    outnames.insert(outnames.begin() + dim, names->begin(), names->end());
    at::internal_set_names_inplace(result, outnames);
  }

  return result;
}

Tensor unflatten(const Tensor& self, int64_t dim, IntArrayRef sizes) {
  return native::unflatten_impl(self, dim, sizes, c10::nullopt);
}

Tensor unflatten(const Tensor& self, Dimname dim, IntArrayRef sizes, DimnameList names) {
  return native::unflatten_impl(self, dimname_to_position(self, dim), sizes, names);
}

Tensor view_as(const Tensor& self, const Tensor& other) {
  return self.view(other.sizes());
}

int64_t numel(const Tensor& self) {
  return self.unsafeGetTensorImpl()->numel();
}

std::vector<Tensor> unbind(const Tensor &self, int64_t dim) {
  dim = maybe_wrap_dim(dim, self.dim());
  int64_t size = self.size(dim);
  std::vector<Tensor> tensors(size);
  for (const auto i : c10::irange(size)) {
    tensors[i] = self.select(dim, i);
  }
  return tensors;
}

std::vector<Tensor> unbind(const Tensor& self, Dimname dim) {
  return at::unbind(self, dimname_to_position(self, dim));
}

std::vector<Tensor> meshgrid(TensorList tensors) {
  TORCH_WARN_ONCE("torch.meshgrid: in an upcoming release, it will be required to pass the "
                  "indexing argument.");
  return native::meshgrid(tensors, /*indexing=*/"ij");
}

std::vector<Tensor> meshgrid(TensorList tensors,
                             c10::string_view indexing) {
  int64_t size = tensors.size();
  TORCH_CHECK(size > 0, "meshgrid expects a non-empty TensorList");

  for(const auto i: c10::irange(size - 1)){
    TORCH_CHECK(tensors[i].dtype() == tensors[i+1].dtype(), "meshgrid expects all tensors to have the same dtype");
    TORCH_CHECK(tensors[i].device() == tensors[i+1].device(), "meshgrid expects all tensors to have the same device");
  }

  // Input tensors is of type TensorList, which is an alias to a
  // constant array slice, which doesn't allow for mutations. We may
  // need to swap our first two elements if indexing is "ij", so we
  // unconditionally create a vector that we can reorder to keep the
  // implementation simple.
  //
  // We are not concerned with the performance of this relative to
  // constructor a grid for each input.
  std::vector<std::reference_wrapper<const Tensor>> tensor_refs(tensors.begin(),
                                                                tensors.end());

  // Whether or not to swap the first two tensors.
  //
  // We only swap if there are at least two* input tensors (obviously)
  // and if indexing is "xy".
  //
  // A reminder about "xy" semantics: "xy" semantics implies that the
  // output grids are in the cartesian coordinate system. Thus the
  // first dimension is the "x" axis (corresponding to column) and the
  // second dimension is the "y" axis (corresponding to row). Tensors,
  // however, generally consider the first axis to be the row and the
  // second axis to be the columns. Thus we flip the two dimensions in
  // contrast to "ij" indexing.
  //
  // It turns out that it's easiest to implement this by just swapping
  // the first two inputs. However, the order of the outputs still
  // must correspond to the order of the inputs. Thus we also must
  // swap the outputs if we swapped the inputs.
  //
  // * Why do we even support this function for exactly one input?
  bool swap_first_and_second_tensors = false;

  if (indexing == "xy") {
    // We can only swap if there are multiple tensors.
    swap_first_and_second_tensors = size >= 2;
    if (swap_first_and_second_tensors) {
      std::swap(tensor_refs[0], tensor_refs[1]);
    }
  } else {
    // Only "xy" and "ij" are supported, and we already checked for
    // "xy" above. Only "ij" remains as a valid mode.
    TORCH_CHECK(indexing == "ij",
                "torch.meshgrid: indexing must be one of \"xy\" or \"ij\", "
                "but received: ", indexing);
  }

  std::vector<int64_t> shape(size);
  for(const auto i: c10::irange(size)){
    TORCH_CHECK(tensor_refs[i].get().dim() <= 1,
                "torch.meshgrid: Expected 0D or 1D tensor in the tensor list but got: ", tensor_refs[i]);
    shape[i] = tensor_refs[i].get().numel();  // treat 0D tensors as if they were a 1D tensor
  }
  std::vector<Tensor> grids;
  std::vector<int64_t> view_shape(size, 1);
  for(const auto i: c10::irange(size)){
    view_shape[i] = -1;  // select this dimension to infer
    grids.push_back(tensor_refs[i].get().view(view_shape).expand(shape));
    view_shape[i] = 1;  // restore to previous value
  }

  // Remember we need to also swap the outputs if we swapped the inputs.
  if (swap_first_and_second_tensors) {
    std::swap(grids[0], grids[1]);
  }
  return grids;
}

// Numpy-style `a.T`: returns the tensor
// with dims reversed
Tensor numpy_T(const Tensor &self) {
  const auto n = self.dim();
  if (n != 2 && n != 0) {
    TORCH_WARN_ONCE(
        "The use of `x.T` on tensors of dimension other than 2 to reverse their shape is deprecated ",
        "and it will throw an error in a future release. Consider `x.mT` to transpose batches of matrices ",
        "or `x.permute(*torch.arange(x.ndim - 1, -1, -1))` to reverse the dimensions of a tensor."
    );
  }
  DimVector transpose_dims;
  for (int64_t i = n - 1; i >= 0; --i) {
    transpose_dims.push_back(i);
  }
  return self.permute(transpose_dims);
}

Tensor matrix_H(const Tensor &self) {
  const auto ndim = self.dim();
  TORCH_CHECK(ndim == 2 || ndim == 0,
      "tensor.H is only supported on matrices (2-D tensors). Got ", ndim, "-D tensor.",
      ndim > 2 ? " For batches of matrices, consider using tensor.mH" : "");
  if (self.is_complex()) {
    return ndim == 0 ? self.conj() : self.transpose(-2, -1).conj();
  } else {
    return ndim == 0 ? self : self.transpose(-2, -1);
  }
}

namespace {
Tensor _adjoint(const Tensor &self, const bool transpose, const char* const name) {
  const auto ndim = self.dim();
  TORCH_CHECK(ndim != 1,
      "tensor.", name, " is only supported on matrices or batches of matrices. Got 1-D tensor.");
  if (transpose || !self.is_complex()) {
    return ndim == 0 ? self : self.transpose(-2, -1);
  } else {
    return ndim == 0 ? self.conj() : self.transpose(-2, -1).conj();
  }
}
} // anonymous namespace

Tensor mT(const Tensor &self) {
  return _adjoint(self, /*transpose=*/true, "mT");
}

Tensor mH(const Tensor &self) {
  return _adjoint(self, /*transpose=*/false, "mH");
}

Tensor adjoint(const Tensor &self) {
  return _adjoint(self, /*transpose=*/false, "adjoint()");
}

Tensor view(const Tensor& self,
            at::IntArrayRef size) {
  return view_impl(self, size);
}

Tensor alias(const Tensor& self) {
  return alias_with_sizes_and_strides(self, self.sizes(), self.strides());
}

Tensor detach(const Tensor& self) {
  // NB: detach() is not the same thing as alias()! The main difference is that
  // detach does not allow metadata change while alias does.
  return Tensor(self.getIntrusivePtr()->shallow_copy_and_detach(
    // NB: The ADInplaceOrView logic will overwrite these with the
    // appropriate values if it runs; otherwise these are the values.
    /*version_counter=*/0,
    /*allow_tensor_metadata_change=*/false));
}

Tensor unfold(const Tensor& self, int64_t d, int64_t size, int64_t step) {
  // some special handling to deal with allow d == 0 when self.dim() == 0
  auto ndim = self.dim();
  d = at::maybe_wrap_dim(d, ndim, /*wrap_scalar=*/true);

  auto sizes = self.sizes().vec();
  auto strides = self.strides().vec();
  int64_t max_size = self.dim() == 0 ? 1 : sizes[d];
  TORCH_CHECK(size <= max_size, "maximum size for tensor at dimension ", d,
                                " is ", max_size, " but size is ", size);
  TORCH_CHECK(step > 0, "step is ", step, " but must be > 0");
  sizes.push_back(size);
  strides.push_back(self.dim() == 0 ? 1 : strides[d]);
  // The if handles the self.dim() == 0 case
  if (d < ndim) {
    sizes[d] = (sizes[d] - size) / step + 1;
    strides[d] *= step;
  }
  return self.as_strided(sizes, strides);
}

Tensor diag(const Tensor& self, int64_t offset) {
  auto ndim = self.dim();
  TORCH_CHECK(ndim == 1 || ndim == 2, "diag(): Supports 1D or 2D tensors. Got ", self.dim(), "D");
  if (ndim == 1) {
    return at::diag_embed(self, offset);
  } else {
    // We return a copy of the diagonal
    return at::diagonal_copy(self, offset);
  }
}

Tensor& diag_out(const Tensor& self, int64_t offset, Tensor& out) {
  auto ndim = self.dim();
  TORCH_CHECK(ndim == 1 || ndim == 2, "Supports 1D or 2D tensors. Got ", self.dim(), "D");
  if (ndim == 1) {
    TORCH_CHECK(
        canCast(self.scalar_type(), out.scalar_type()),
        "diag: result type ", self.scalar_type(), " can't be cast to the desired out= type ",
        out.scalar_type());
    return at::diag_embed_out(out, self, offset);
  } else {
    return at::diagonal_copy_out(out, self, offset);
  }
}

Tensor diagonal_backward_symint(const Tensor & grad, SymIntArrayRef input_sizes, int64_t offset, int64_t dim1, int64_t dim2) {
  auto grad_input = at::zeros_symint(input_sizes, grad.options());
  auto diag = grad_input.diagonal(offset, dim1, dim2);
  diag.copy_(grad);
  return grad_input;
}

Tensor movedim(const Tensor& self, IntArrayRef src, IntArrayRef dst) {
  TORCH_CHECK(src.size() == dst.size(), "movedim: Invalid source or destination dims: source (",
              src, " dims) should contain the same number of dims as destination (", dst, " dims)");

  size_t self_dim = self.dim();
  DimVector normalized_src(src.size());
  DimVector normalized_dst(dst.size());

  auto wrap_dims = [&self_dim](const IntArrayRef& vec, DimVector& normalized_vec) {
    for (const auto i : c10::irange(vec.size())) {
      normalized_vec[i] = maybe_wrap_dim(vec[i], self_dim);
    }
  };

  wrap_dims(src, normalized_src);
  wrap_dims(dst, normalized_dst);

  auto all_unique = [](const DimVector& dims) {
    DimVector copy = dims;
    std::sort(copy.begin(), copy.end());
    auto duplicate = std::adjacent_find(copy.begin(), copy.end());
    return duplicate == copy.end();
  };
  TORCH_CHECK(all_unique(normalized_src), "movedim: repeated dim in `source` (", src, ")");
  TORCH_CHECK(all_unique(normalized_dst), "movedim: repeated dim in `destination` (", dst, ")");

  // handle the case of scalar tensor as a no-op
  if (self_dim == 0)
    return self.alias();

  // TODO: The algorithm below can probably be optimized.
  // Reference: https://github.com/pytorch/pytorch/pull/41480#discussion_r456100505

  // Algorithm Walkthrough
  // Example Input
  // Variable State:
  //     normalized_src = 0, 1
  //     normalized_dst = 2, 4
  //     self_dim = 5
  DimVector order(self_dim);
  DimVector source_dims(self_dim);
  DimVector destination_dims(self_dim);

  // We initialize two vectors to track update to the dims
  // `order` contains the final order of the dim positions.
  // Variable State:
  //     order = NA, NA, NA, NA, NA
  //     source_dims = 0, 1, 2, 3, 4
  //     destination_dims = 0, 1, 2, 3, 4
  std::iota(source_dims.begin(), source_dims.end(), 0);
  std::iota(destination_dims.begin(), destination_dims.end(), 0);

  // We mark and update position for the dim provided by user
  // i.e. `normalized_src` and `normalized_dims`
  // Variable State:
  //     order = NA, NA, 0, NA, 1
  //     source_dims = -1, -1, 2, 3, 4
  //     destination_dims = 0, 1, -1, 3, -1
  for (const auto i : c10::irange(src.size())) {
      order[normalized_dst[i]] = normalized_src[i];
      source_dims[normalized_src[i]] = -1;
      destination_dims[normalized_dst[i]] = -1;
  }

  // Remove the dims whose position we already know,
  // the ones marked with -1 in previous step
  // Variable State:
  //     source_dims = 2, 3, 4
  //     destination_dims = 0, 1, 3
  auto source_iter = std::remove(source_dims.begin(), source_dims.end(), -1);
  auto destination_iter = std::remove(destination_dims.begin(), destination_dims.end(), -1);

  int64_t rest_dim = self.dim() - src.size();
  TORCH_INTERNAL_ASSERT(std::distance(source_dims.begin(), source_iter)  == rest_dim);
  TORCH_INTERNAL_ASSERT(std::distance(destination_dims.begin(), destination_iter)  == rest_dim);

  // Update the position of the remaining dimensions.
  // `source_dims` now contains the original position
  // `destination_dims` contains the new position it will shifted to
  // after considering the user inputs.
  // Variable State:
  //     order = 2, 3, 0, 4, 1
  for (const auto i : c10::irange(rest_dim)) {
      order[destination_dims[i]] = source_dims[i];
  }

  return self.permute(order);
}

Tensor movedim(const Tensor& self, int64_t src, int64_t dst) {
  return at::movedim(self, IntArrayRef{src}, IntArrayRef{dst});
}

Tensor moveaxis(const Tensor& self, IntArrayRef src, IntArrayRef dst) {
  return at::movedim(self, src, dst);
}

Tensor moveaxis(const Tensor& self, int64_t src, int64_t dst) {
  return at::movedim(self, IntArrayRef{src}, IntArrayRef{dst});
}

Tensor swapaxes(const Tensor& self, int64_t axis0, int64_t axis1) {
  return self.transpose(axis0, axis1);
}

Tensor& swapaxes_(Tensor& self, int64_t axis0, int64_t axis1) {
  return self.transpose_(axis0, axis1);
}

Tensor swapdims(const Tensor& self, int64_t dim0, int64_t dim1) {
  return self.transpose(dim0, dim1);
}

Tensor& swapdims_(Tensor& self, int64_t dim0, int64_t dim1) {
  return self.transpose_(dim0, dim1);
}

Tensor flatten_dense_tensors(TensorList tensors) {
  static auto flatten = [](const Tensor &t) { return t.contiguous().view({-1}); };
  if (tensors.size() == 1)
    return flatten(tensors[0]);
  return at::cat(fmap(tensors, flatten));
}

std::vector<Tensor> unflatten_dense_tensors(const Tensor& flat, TensorList tensors) {
  std::vector<Tensor> outputs;
  outputs.reserve(tensors.size());
  size_t offset = 0;
  for (const auto & tensor : tensors) {
    auto numel = tensor.numel();
    // If unflatten an empty tensor, create a new empty tensor using
    // flat tensor Options.
    // This can avoid the unflattened empty tensor to share the same storage
    // with other unflatten tensors.
    if (numel == 0) {
      outputs.push_back(at::empty({0}, flat.options()));
    } else {
      outputs.push_back(flat.narrow(0, offset, numel).view(tensor.sizes()));
      offset += numel;
    }
  }
  return outputs;
}

at::Tensor slice_scatter(const at::Tensor& self, const at::Tensor& src, int64_t dim, c10::optional<int64_t> start, c10::optional<int64_t> end, int64_t step) {
    auto output = self.clone();
    auto slice = output.slice(dim, start, end, step);
    TORCH_CHECK(slice.sizes() == src.sizes(), "expected src to have a size equal to the slice of self. src size = ", src.sizes(), ", slice size = ", slice.sizes());
    slice.copy_(src);
    return output;
}
at::Tensor select_scatter(const at::Tensor& self, const at::Tensor& src, int64_t dim, int64_t index) {
    auto output = self.clone();
    auto slice = output.select(dim, index);
    TORCH_CHECK(slice.sizes() == src.sizes(), "expected src to have a size equal to the slice of self. src size = ", src.sizes(), ", slice size = ", slice.sizes());
    slice.copy_(src);
    return output;
}
at::Tensor diagonal_scatter(const at::Tensor& self, const at::Tensor& src, int64_t offset, int64_t dim1, int64_t dim2) {
    auto output = self.clone();
    auto slice = output.diagonal(offset, dim1, dim2);
    TORCH_CHECK(slice.sizes() == src.sizes(), "expected src to have a size equal to the slice of self. src size = ", src.sizes(), ", slice size = ", slice.sizes());
    slice.copy_(src);
    return output;
}
at::Tensor as_strided_scatter_symint(const at::Tensor& self, const at::Tensor& src, at::SymIntArrayRef size, at::SymIntArrayRef stride, c10::optional<c10::SymInt> storage_offset) {
    // See Note [as_strided_scatter backward support]
    TORCH_INTERNAL_ASSERT(!self.requires_grad() || self.is_contiguous(), "as_strided_scatter is currently only supported for contiguous inputs");
    auto output = self.clone();
    auto slice = output.as_strided_symint(size, stride, storage_offset);
    TORCH_CHECK(slice.sym_sizes() == src.sym_sizes(), "expected src to have a size equal to the slice of self. src size = ", src.sym_sizes(), ", slice size = ", slice.sym_sizes());
    slice.copy_(src);
    return output;
}

// The default implementation of lift is a no-op.
// If TLS is set appropriately (for wrapper-tensor keys like Functionalize or functorch transforms),
// then we'll dispatch to one of their implementations, which will properly lift the tensor into a wrapper.
at::Tensor lift(const at::Tensor& self) {
    return self;
}

// See notes in native_functions.yaml
at::Tensor lift_fresh(const at::Tensor& self) {
    return self;
}

at::Tensor& _fw_primal_copy_out(const at::Tensor & self, int64_t level, at::Tensor & out) {
  auto tmp = self._fw_primal(level);
  out.copy_(tmp);
  return out;
}


at::Tensor& _make_dual_copy_out(const at::Tensor & primal, const at::Tensor & tangent, int64_t level, at::Tensor & out) {
  auto tmp = at::_make_dual(primal, tangent, level);
  out.copy_(tmp);
  return out;
}


at::Tensor& view_as_real_copy_out(const at::Tensor & self, at::Tensor & out) {
  auto tmp = at::view_as_real(self);
  out.copy_(tmp);
  return out;
}


at::Tensor& view_as_complex_copy_out(const at::Tensor & self, at::Tensor & out) {
  auto tmp = at::view_as_complex(self);
  out.copy_(tmp);
  return out;
}


at::Tensor& _conj_copy_out(const at::Tensor & self, at::Tensor & out) {
  auto tmp = self._conj();
  out.copy_(tmp);
  return out;
}


at::Tensor& _neg_view_copy_out(const at::Tensor & self, at::Tensor & out) {
  auto tmp = self._neg_view();
  out.copy_(tmp);
  return out;
}


at::Tensor& as_strided_copy_out_symint(const at::Tensor & self, at::SymIntArrayRef size, at::SymIntArrayRef stride, c10::optional<c10::SymInt> storage_offset, at::Tensor & out) {
  auto tmp = self.as_strided_symint(size, stride, storage_offset);
  out.copy_(tmp);
  return out;
}


at::Tensor& _sparse_broadcast_to_copy_out(const at::Tensor & self, at::IntArrayRef size, at::Tensor & out) {
  auto tmp = at::_sparse_broadcast_to(self, size);
  out.copy_(tmp);
  return out;
}


at::Tensor& diagonal_copy_out(const at::Tensor & self, int64_t offset, int64_t dim1, int64_t dim2, at::Tensor & out) {
  TORCH_CHECK(
    out.device() == self.device(),
    "diagonal_copy: Expected out and self tensors to be on the same device, but got ",
    "out on ", out.device(), " and self on ", self.device());
  auto result = self.diagonal(offset, dim1, dim2);
  at::native::resize_output(out, result.sizes());
  TORCH_CHECK(
      canCast(result.scalar_type(), out.scalar_type()),
      "diagonal_copy: result type ", result.scalar_type(), " can't be cast to the desired out= type ", out.scalar_type());
  out.copy_(result);
  return out;
}


at::Tensor& expand_copy_SymInt_out(const at::Tensor & self, c10::SymIntArrayRef size, bool implicit, at::Tensor & out) {
  auto tmp = self.expand_symint(size, implicit);
  out.copy_(tmp);
  return out;
}


at::Tensor& expand_copy_out_symint(const at::Tensor & self, at::SymIntArrayRef size, bool implicit, at::Tensor & out) {
  auto tmp = self.expand_symint(size, implicit);
  out.copy_(tmp);
  return out;
}


at::Tensor& narrow_copy_out(const at::Tensor & self, int64_t dim, int64_t start, int64_t length, at::Tensor & out) {
  auto tmp = self.narrow(dim, start, length);
  out.copy_(tmp);
  return out;
}


at::Tensor& permute_copy_out(const at::Tensor & self, at::IntArrayRef dims, at::Tensor & out) {
  auto tmp = self.permute(dims);
  out.copy_(tmp);
  return out;
}


at::Tensor& _reshape_alias_copy_out(const at::Tensor & self, at::IntArrayRef size, at::IntArrayRef stride, at::Tensor & out) {
  auto tmp = self._reshape_alias(size, stride);
  out.copy_(tmp);
  return out;
}


at::Tensor& select_copy_int_out(const at::Tensor & self, int64_t dim, int64_t index, at::Tensor & out) {
  auto tmp = self.select(dim, index);
  out.copy_(tmp);
  return out;
}


at::Tensor& detach_copy_out(const at::Tensor & self, at::Tensor & out) {
  auto tmp = self.detach();
  out.copy_(tmp);
  return out;
}


at::Tensor& slice_copy_Tensor_out(const at::Tensor & self, int64_t dim, c10::optional<int64_t> start, c10::optional<int64_t> end, int64_t step, at::Tensor & out) {
  auto tmp = self.slice(dim, start, end, step);
  out.copy_(tmp);
  return out;
}


void split_copy_Tensor_out(const at::Tensor & self, int64_t split_size, int64_t dim, at::TensorList  out) {
  auto tmp = self.split(split_size, dim);

  TORCH_CHECK(out.size() == tmp.size(), "split_copy_Tensor_out() expected an out= argument of size ", tmp.size(), ", got size ", out.size());
  for (const auto i : c10::irange(out.size())) {
    out[i].copy_(tmp[i]);
  }
}


void split_with_sizes_copy_out(const at::Tensor & self, at::IntArrayRef split_sizes, int64_t dim, at::TensorList  out) {
  auto tmp = self.split_with_sizes(split_sizes, dim);

  TORCH_CHECK(out.size() == tmp.size(), "split_with_sizes_copy_out() expected an out= argument of size ", tmp.size(), ", got size ", out.size());
  for (const auto i : c10::irange(out.size())) {
    out[i].copy_(tmp[i]);
  }
}


at::Tensor& squeeze_copy_out(const at::Tensor & self, at::Tensor & out) {
  auto tmp = self.squeeze();
  out.copy_(tmp);
  return out;
}


at::Tensor& squeeze_copy_dim_out(const at::Tensor & self, int64_t dim, at::Tensor & out) {
  auto tmp = self.squeeze(dim);
  out.copy_(tmp);
  return out;
}


at::Tensor& t_copy_out(const at::Tensor & self, at::Tensor & out) {
  auto tmp = self.t();
  out.copy_(tmp);
  return out;
}


at::Tensor& transpose_copy_int_out(const at::Tensor & self, int64_t dim0, int64_t dim1, at::Tensor & out) {
  auto tmp = self.transpose(dim0, dim1);
  out.copy_(tmp);
  return out;
}


at::Tensor& unsqueeze_copy_out(const at::Tensor & self, int64_t dim, at::Tensor & out) {
  auto tmp = self.unsqueeze(dim);
  out.copy_(tmp);
  return out;
}


at::Tensor& _indices_copy_out(const at::Tensor & self, at::Tensor & out) {
  auto tmp = self._indices();
  out.copy_(tmp);
  return out;
}


at::Tensor& _values_copy_out(const at::Tensor & self, at::Tensor & out) {
  auto tmp = self._values();
  out.copy_(tmp);
  return out;
}


at::Tensor& indices_copy_out(const at::Tensor & self, at::Tensor & out) {
  auto tmp = self.indices();
  out.copy_(tmp);
  return out;
}


at::Tensor& values_copy_out(const at::Tensor & self, at::Tensor & out) {
  auto tmp = self.values();
  out.copy_(tmp);
  return out;
}


at::Tensor& crow_indices_copy_out(const at::Tensor & self, at::Tensor & out) {
  auto tmp = self.crow_indices();
  out.copy_(tmp);
  return out;
}


at::Tensor& col_indices_copy_out(const at::Tensor & self, at::Tensor & out) {
  auto tmp = self.col_indices();
  out.copy_(tmp);
  return out;
}


void unbind_copy_int_out(const at::Tensor & self, int64_t dim, at::TensorList  out) {
  auto tmp = self.unbind(dim);

  TORCH_CHECK(out.size() == tmp.size(), "unbind_copy_int_out() expected an out= argument of size ", tmp.size(), ", got size ", out.size());
  for (const auto i : c10::irange(out.size())) {
    out[i].copy_(tmp[i]);
  }
}


at::Tensor& view_copy_out_symint(const at::Tensor & self, at::SymIntArrayRef size, at::Tensor & out) {
  auto tmp = self.view_symint(size);
  out.copy_(tmp);
  return out;
}


at::Tensor& view_copy_dtype_out(const at::Tensor & self, at::ScalarType dtype, at::Tensor & out) {
  auto tmp = self.view(dtype);
  out.copy_(tmp);
  return out;
}


at::Tensor& unfold_copy_out(const at::Tensor & self, int64_t dimension, int64_t size, int64_t step, at::Tensor & out) {
  auto tmp = self.unfold(dimension, size, step);
  out.copy_(tmp);
  return out;
}


at::Tensor& alias_copy_out(const at::Tensor & self, at::Tensor & out) {
  auto tmp = self.alias();
  out.copy_(tmp);
  return out;
}

int64_t sparse_dim_strided(const at::Tensor& self) {
  return 0;
}

int64_t dense_dim_strided(const at::Tensor& self) {
  return self.dim();
}

} // namespace native
} // namespace at
