// #define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <c10/util/Optional.h>
#include <ATen/quantized/Quantizer.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorOperators.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_autocast_to_full_precision_native.h>
#include <ATen/ops/_autocast_to_reduced_precision_native.h>
#include <ATen/ops/_convert_indices_from_coo_to_csr.h>
#include <ATen/ops/_convert_indices_from_coo_to_csr_native.h>
#include <ATen/ops/_convert_indices_from_csr_to_coo.h>
#include <ATen/ops/_convert_indices_from_csr_to_coo_native.h>
#include <ATen/ops/_sparse_bsc_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_bsr_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_compressed_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_coo_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_csc_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_csr_tensor_unsafe_native.h>
#include <ATen/ops/_to_copy.h>
#include <ATen/ops/_to_copy_native.h>
#include <ATen/ops/_to_cpu_native.h>
#include <ATen/ops/_to_dense_native.h>
#include <ATen/ops/arange_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/empty_quantized.h>
#include <ATen/ops/empty_strided.h>
#include <ATen/ops/empty_strided_native.h>
#include <ATen/ops/to_dense_backward_native.h>
#include <ATen/ops/to_dense_native.h>
#include <ATen/ops/to_mkldnn_backward_native.h>
#include <ATen/ops/to_native.h>
#include <ATen/ops/to_sparse_bsc_native.h>
#include <ATen/ops/to_sparse_bsr_native.h>
#include <ATen/ops/to_sparse_csc_native.h>
#include <ATen/ops/to_sparse_csr_native.h>
#include <ATen/ops/to_sparse_native.h>
#include <ATen/ops/view_native.h>
#include <ATen/ops/zeros.h>
#endif

#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/SparseTensorUtils.h>
#include <ATen/core/ATen_fwd.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/NonSymbolicBC.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <numeric>

namespace at {
namespace native {

namespace {
// dense_to_sparse_{csr,bsr,csc,bsc} common helpers

// Preparation fo the N-D dense -> sparse compressed conversion.
// The N-D input is converted to 3-D (single batch dim) where we check that the
// product of batch dims is nonzero and for each batch the sparse matrix
// contained within has the same number of non-zero elements.
// The batches are joined along the compressed axis. The generation of indices
// for this matrix can be performed in a single step followed by a single step
// conversion to restore the batch dimension.
void dense_to_sparse_compressed_prepare_check_mask_values_batched(
    const Layout& target_layout,
    Tensor& values,
    Tensor& mask,
    const int64_t& n_batch_dim) {
  if (n_batch_dim > 1) {
    // For inputs with more than 1 batch dim we flatten them out.
    // Input shape (b0, b1 ..., bn, r, c) -> (b0 * b1 * ... * bn, r ,c)
    values = values.flatten(0, n_batch_dim - 1);
    mask = mask.flatten(0, n_batch_dim - 1);
  }

  // For informative messaging form the name of the function
  // to_sparse_{csr,csc,bsr,bsc}.
  TORCH_CHECK(
      mask.size(0) > 0,
      "to_sparse_",
      // We want the message to match the function name so generate the
      // lowercase acronym for the layout
      sparse_csr::layoutToString(target_layout, false, true),
      ": Expected product of batch dimensions to be non-zero.");

  // Compute the number of non-zero elements in the first batch, expand to full
  // size
  auto nse_per_batch = mask.select(0, 0).sum().expand(mask.size(0));
  TORCH_CHECK(
      mask.sum({-2, -1}).equal(nse_per_batch),
      "Expect the same number of specified elements per batch.");

  // We need to join batches into a matrix increasing the length of the
  // compressed axis. This allows us to create indices for a compressed matrix
  // and de-batch them later (two kernels). Otherwise we would have to create
  // indices for each batch individually requiring n_batch kernels. For csr/bsr,
  // we already have the batch dim adjacent to the compressed axis and can
  // flatten them together. For csc/bsc, we need to transpose first.
  // For BSR/CSR (b, r, c) -> (b*r, c)
  // For BSC/CSC (b, c, r) -> (r, b*c)
  AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(
      target_layout,
      "dense_to_sparse_compressed",
      [&]() {
        values = values.flatten(0, 1);
        mask = mask.flatten(0, 1);
      },
      [&]() {
        values = values.transpose(0, 1).flatten(1, 2);
        mask = mask.transpose(0, 1).flatten(1, 2);
      });
}

// This function unfolds the compressed indices of a compressed sparse matrix
// into a batched compressed sparse tensor.
// This is analogous to an unflatten-like operation:
// unflatten(0, {b, r}) for csr/bsr with input shape (r*b, c)
//          (output shape (b, r, c))
// unflatten(1, {b, c}).transpose(0,1) for csc/bsc with input shape (r, c*b)
//          (output shape (r, b, c) unflatten, (b, r, c) unflatten + transpose)
// This only operates on the compressed indices as the plain indices and values
// can be manipulated as described above without special handling.
// It is a prerequisite for the conversion that the sparsity pattern is sane for
// the batched shape. That is each batch has the same number of nonzero
// elements.
Tensor compressed_to_batched_compressed_indices(
    const Tensor& compressed_in,
    const int64_t& n_batch,
    bool out_int32) {
  auto n_compressed_per_batch = (compressed_in.size(0) - 1) / n_batch;
  ScalarType out_type = out_int32 ? ScalarType::Int : ScalarType::Long;
  auto batched_out = at::zeros(
      {n_batch, n_compressed_per_batch + 1},
      compressed_in.options().dtype(out_type));

  // If the compressed dimension has length zero there is 1 element in each
  // batch and it is zero we already have this result formed
  if (n_compressed_per_batch > 0) {
    // Slice the compressed indices ignoring the leading 0 element and reshape
    // to n-batch rows
    auto trailing_slice =
        compressed_in.slice(0, 1, c10::nullopt, 1).reshape({n_batch, -1});
    // Slice the compressed indices again selecting the elements corresponding
    // to the batch boundary. The values here will be increasing multiples of
    // nnz per batch. Reshape to n-batch rows (1 col) for broadcasting.
    // This is equivalent to arange(n_batch) * nnz_per_batch with the same
    // reshape
    auto offsets = compressed_in.slice(0, 0, -1, n_compressed_per_batch)
                       .reshape({n_batch, -1});
    // Subtracting the offsets from each row of the reshaped compressed indices
    // gives us the compressed indices within the batch. The leading element of
    // each row is not computed as it is always zero.  We copy into the view on
    // the output buffer.
    batched_out.narrow(-1, 1, n_compressed_per_batch)
        .copy_(trailing_slice - offsets);
  }
  return batched_out;
}

// After generating member tensors for sparse_compressed matrix, if the target
// shape is N-D we must reform the batch dimensions.
// Single kernel is used to restore one batch dimension in the compressed
// indices. From there full batch shape is restored by reshape. No special
// handling is needed for restoring batch dimensions of the values or
// plain_indices it can be done with reshape/unflatten.
void reshape_2d_sparse_compressed_members_to_nd_batched(
    const IntArrayRef full_sizes,
    const int64_t& n_batch_dim,
    Tensor& compressed_indices,
    Tensor& plain_indices,
    Tensor& values) {
  auto batch_shape = full_sizes.slice(0, n_batch_dim);
  auto n_batch = std::accumulate(
      batch_shape.begin(), batch_shape.end(), 1, std::multiplies<int64_t>());
  // NOTE: using this conversion requires the nnz per batch is the same for all
  // batches that will be formed. We ensured this was the case on the way in so
  // it is safe to use this conversion.
  compressed_indices = compressed_to_batched_compressed_indices(
      compressed_indices, n_batch, /*out_int32*/ false);

  // We can infer the last dim of the reshape targets, it will be nnz or
  // nrow/ncol+1 depending on the layout and member tensor targeted.
  auto batchsize_infer_last = DimVector(batch_shape);
  batchsize_infer_last.push_back(-1);

  // -1 will be nnz per batch
  plain_indices = plain_indices.reshape(batchsize_infer_last);
  // -1 will be ncols (bsc,csc) or nrows (bsr,csr) + 1
  compressed_indices = compressed_indices.reshape(batchsize_infer_last);
  // -1 will be nnz (per batch).
  // Note: Unflatten rather than reshape as it will work
  // for both blocked and unblocked layouts. reshape works for unblocked layouts
  // only
  values = values.unflatten(0, batchsize_infer_last);
}
} // namespace

// Take a Device that may not have device_index set (i.e., having it as -1
// representing the current device) and return the corresponding Device
// according to the actual device at the time of this function call.  No-op
// if the device_index is set.
static inline Device ensure_has_index(Device device) {
  if (device.is_cpu() || device.has_index()) {
    return device;
  }
  const c10::impl::DeviceGuardImplInterface* impl = c10::impl::getDeviceGuardImpl(device.type());
  return impl->getDevice();
}

static inline optional<Device> ensure_has_index(optional<Device> device) {
  if (!device.has_value()) {
    return nullopt;
  }
  return ensure_has_index(device.value());
}

Tensor _to_copy(
    const Tensor& self,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory,
    bool non_blocking,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  TORCH_CHECK(!layout.has_value() || self.layout() == layout.value(),
           "to(options) doesn't support converting to a different layout, "
           "but got self.layout being ", self.layout(),
           " and options.layout set as ", layout.value());
  auto options = TensorOptions()
    .dtype(dtype)
    .layout(layout)
    .device(device)
    .pinned_memory(pin_memory);

  if (options.has_device()) {
    options = options.device(ensure_has_index(options.device()));
  }
  // memory_format is handled separately due to MemoryFormat::Preserve logic
  options = self.options().merge_in(options).memory_format(c10::nullopt);
  auto memory_format = optional_memory_format.value_or(MemoryFormat::Preserve);

  // TODO: Use the dispatcher for this.
  // Currently there are unenumerated extensibility issues preventing this.
  if (at::sparse_csr::is_sparse_compressed(self)) {
      TORCH_CHECK(
          memory_format == MemoryFormat::Preserve,
          "to(options): ", at::sparse_csr::layoutToString(self.layout()),
          " only supports memory format Preserve, but got ", memory_format,
          " instead.");

      Tensor compressed_indices, plain_indices;
      std::tie(compressed_indices, plain_indices) = at::sparse_csr::getCompressedPlainIndices(self);

      const auto new_values = at::native::to(
          self.values(),
          dtype,
          c10::kStrided,
          device,
          pin_memory,
          non_blocking,
          true, // force copy since we are in _to_copy
          memory_format);

      const auto new_compressed_indices = at::native::to(
          compressed_indices,
          compressed_indices.scalar_type(),
          c10::kStrided,
          device,
          pin_memory,
          non_blocking,
          true, // force copy since we are in _to_copy
          memory_format);

      const auto new_plain_indices = at::native::to(
          plain_indices,
          plain_indices.scalar_type(),
          c10::kStrided,
          device,
          pin_memory,
          non_blocking,
          true, // force copy since we are in _to_copy
          memory_format);

    return at::native::_sparse_compressed_tensor_unsafe(
        new_compressed_indices,
        new_plain_indices,
        new_values,
        self.sizes(),
        new_values.scalar_type(),
        self.layout(),
        new_values.device());
  }

  bool pin_out = (non_blocking && self.is_cuda() && options.device().is_cpu() &&
                  (options.layout() == c10::kStrided));

  if (memory_format == MemoryFormat::Preserve) {
    if (options.device().supports_as_strided()) {
      if (self.is_non_overlapping_and_dense()) {
        Tensor r;
        if (self.is_quantized()) {
          r = at::empty_quantized(self.sizes(), self, options);
          at::QuantizerPtr quantizer = r.quantizer();
          r.copy_(self, non_blocking);
          set_quantizer_(r, quantizer);
        } else {
          r = at::empty_strided(
              self.sizes(),
              self.strides(),
              options.pinned_memory(pin_out));
          r.copy_(self, non_blocking);
        }
        return r;
      } else if (!self.is_quantized() && self.layout() == kStrided) {
          Tensor r;
          auto strides = infer_dense_strides(self.sizes(), self.strides());
          r = at::empty_strided(
              self.sizes(),
              strides,
              options.pinned_memory(pin_out));
          r.copy_(self, non_blocking);
          return r;
      } else {
        memory_format = self.suggest_memory_format();
      }
    } else {
      memory_format = self.suggest_memory_format();
    }
  }
  // See Note [Explicit nullopt MemoryFormat argument]
  // TODO: empty_quantized does not work here. It raises an exception in CheckMemoryFormat.h prior to
  // empty_affine_quantizd/_empty_per_channel_affine_quantized calls
  // at::empty also does not work here because there is no proper at::empty support for quantized tensors
  // as it would return a quantized tensor with an UnknownQuantizer
  auto r = self.is_quantized() ? at::empty_like(self, memory_format)
                               : at::empty(self.sizes(),
                                 options.memory_format(memory_format).pinned_memory(pin_out), c10::nullopt);
  r.copy_(self, non_blocking);
  return r;
}

template <typename T>
static inline bool is_null_or_equal_to(const c10::optional<T>& test, const T& value) {
  if (!test.has_value()) {
    return true;
  }
  return test.value() == value;
}

// NOTE: static runtime's to_maybe_copy_out relies on details of this
// check; if you change how it works, please update static runtime as
// well.
bool to_will_alias(
    const Tensor& self,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    bool copy,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  auto memory_format = optional_memory_format.value_or(MemoryFormat::Preserve);

  return is_null_or_equal_to(dtype, self.dtype().toScalarType()) &&
    is_null_or_equal_to(layout, self.layout()) &&
    is_null_or_equal_to(device, self.device()) &&
    !copy &&
    (memory_format == MemoryFormat::Preserve ||
     self.suggest_memory_format() == memory_format);
}

static inline Tensor to_impl(
    const Tensor& self,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory,
    bool non_blocking,
    bool copy,
    c10::optional<c10::MemoryFormat> optional_memory_format) {

  // fast path
  if (to_will_alias(self, dtype, layout, device, copy, optional_memory_format)) {
    return self;
  }
  return at::_to_copy(
      self, dtype, layout, device, pin_memory, non_blocking, optional_memory_format);
}

// If input tensor is fp32, cast it to fp16, otherwise leave it alone.
// (this is intended to be used internally by the JIT autocast implementation)
Tensor _autocast_to_reduced_precision(const Tensor& self, bool cuda_enabled, bool cpu_enabled, ScalarType cuda_dtype, ScalarType cpu_dtype) {
  if (self.dtype() == at::ScalarType::Float &&
      ((self.device().is_cuda() && cuda_enabled) ||
      (self.device().is_cpu() && cpu_enabled))
      ) {
    at::ScalarType target = at::ScalarType::Undefined;
    if (self.device().is_cuda()) {
      target = cuda_dtype;
    } else if (self.device().is_cpu()) {
      target = cpu_dtype;
    }

    TORCH_INTERNAL_ASSERT(target != at::ScalarType::Undefined, "_autocast_to_reduced_precision requires legit ScalarType argument for given device");

    return to_impl(
        self, target, c10::nullopt, c10::nullopt, c10::nullopt, false, false, c10::nullopt);
  } else {
    return self;
  }
}

// If input tensor is fp16, cast it to fp32, otherwise leave it alone.
// (this is intended to be used internally by the JIT autocast implementation)
Tensor _autocast_to_full_precision(const Tensor& self, bool cuda_enabled, bool cpu_enabled) {
  if ((self.dtype() == at::ScalarType::Half || self.dtype() == at::ScalarType::BFloat16) &&
      ((self.device().is_cuda() && cuda_enabled) ||
      (self.device().is_cpu() && cpu_enabled))
      ) {
    return to_impl(
        self, at::ScalarType::Float, c10::nullopt, c10::nullopt, c10::nullopt, false, false, c10::nullopt);
  } else {
    return self;
  }
}

Tensor to(
  const Tensor& self,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory,
  bool non_blocking,
  bool copy,
  c10::optional<c10::MemoryFormat> optional_memory_format
) {
  return to_impl(
      self,
      dtype,
      layout,
      ensure_has_index(device),
      pin_memory,
      non_blocking,
      copy,
      optional_memory_format);
}

Tensor to(const Tensor& self, Device device, ScalarType dtype, bool non_blocking, bool copy, c10::optional<c10::MemoryFormat> optional_memory_format) {
  return to_impl(
      self,
      dtype,
      nullopt,
      ensure_has_index(device),
      nullopt,
      non_blocking,
      copy,
      optional_memory_format);
}

Tensor to(const Tensor& self, ScalarType dtype, bool non_blocking, bool copy, c10::optional<c10::MemoryFormat> optional_memory_format) {
  return to_impl(
      self,
      dtype,
      nullopt,
      nullopt,
      nullopt,
      non_blocking,
      copy,
      optional_memory_format);
}

Tensor to(const Tensor& self, const Tensor& other, bool non_blocking, bool copy, c10::optional<c10::MemoryFormat> optional_memory_format) {
  auto options = other.options();
  return to_impl(
      self,
      options.dtype().toScalarType(),
      options.layout(),
      options.device(),
      options.pinned_memory(),
      non_blocking,
      copy,
      optional_memory_format);
}

// This op is important primarily for lazy / graph-based backends.
// While this vanilla implementation loops through each tensor and independently converts it to cpu,
// a lazy backend like XLA might need to tell sync updates across tensors.
std::vector<Tensor> _to_cpu(TensorList tensors) {
    std::vector<Tensor> cpu_tensors;
    for (const auto& t : tensors) {
        cpu_tensors.push_back(t.cpu());
    }
    return cpu_tensors;
}

Tensor to_dense_backward(const Tensor& grad, const Tensor& input_) {
  AT_ASSERT(input_.layout() != c10::kStrided);
  if (input_.layout() == c10::kSparse) {
    auto input = input_.coalesce();
    return grad.sparse_mask(input);
  }
  if (at::sparse_csr::is_sparse_compressed(input_)) {
    // TODO: implement sparse_compressed_mask
    switch(input_.layout()) {
    case kSparseCsr: return grad.sparse_mask(input_.to_sparse()).to_sparse_csr();
    case kSparseCsc: return grad.sparse_mask(input_.to_sparse()).to_sparse_csc();
      // BSR and BSC should be handled via implement sparse_compressed_mask
    default: ; // fall back to unsupported input layout error
    }
  }
  if (input_.layout() == c10::kMkldnn) {
    return grad.to_mkldnn(input_.scalar_type());
  }
  if (input_.layout() == c10::kStrided) {
    return grad.to_dense();
  }
  AT_ERROR("to_dense_backward: Unsupported input layout: ", input_.layout());
}

Tensor to_mkldnn_backward(const Tensor& grad, const Tensor& input_) {
  AT_ASSERT(input_.layout() == c10::kStrided);
  return grad.to_dense(input_.scalar_type());
}

Tensor to_dense(const Tensor& tensor, c10::optional<c10::ScalarType> dtype) {
  if (tensor.layout() == c10::kSparse) {
    return tensor._to_dense(dtype);
  }
  if (tensor.layout() == c10::kSparseCsr ||
      tensor.layout() == c10::kSparseCsc ||
      tensor.layout() == c10::kSparseBsr ||
      tensor.layout() == c10::kSparseBsc) {
    return tensor._to_dense(dtype);
  }
  if (tensor.layout() == c10::kMkldnn) {
    return tensor._to_dense(dtype);
  }
  TORCH_CHECK(
      tensor.layout() == c10::kStrided,
      "to_dense does not support layout ",
      tensor.layout());
  if (dtype) {
    return tensor.to(*dtype);
  }
  return tensor;
}

Tensor sparse_to_dense(const Tensor& self, c10::optional<ScalarType> dtype) {
  TORCH_CHECK(
      !dtype.has_value(), "dtype argument is not supported by sparse_to_dense");
  Tensor dst = at::zeros(self.sizes(), self.options().layout(kStrided));
  return dst.add_(self);
}

Tensor sparse_compressed_to_dense(
    const Tensor& self,
    c10::optional<ScalarType> dtype) {
  TORCH_CHECK(
      !dtype.has_value(),
      "dtype argument is not supported by sparse_csr_to_dense");

  // Guard upfront against hybrid tensors (causes segfault)
  auto batch_ndim = sparse_csr::numBatchDimensions(self);

  TORCH_CHECK(
      (self.dim() - batch_ndim) == 2,
      "sparse_compressed_to_dense: Hybrid tensors are not supported");

  if (self.layout() == kSparseCsr) {
    Tensor dst = at::zeros(self.sizes(), self.options().layout(kStrided));
    return dst.add_(self);
  }
  // dense.add_ is not yet implemented for CSC.
  // Once it is there, use add_ instead.
  // It is easier to implement it like this for now,
  // because add_ will be modified to work with
  // dense dimensions once CSR/CSC support them.
  if (self.layout() == kSparseCsc) {
    const auto batch_ndim = self.ccol_indices().dim() - 1;
    auto dst_transposed_sizes = self.sizes().vec();
    std::swap(dst_transposed_sizes[batch_ndim], dst_transposed_sizes[batch_ndim + 1]);
    // TODO: write a utility function, or use a transpose once view semantics are there
    const auto to_transposed_csr = at::native::_sparse_csr_tensor_unsafe(
        self.ccol_indices(),
        self.row_indices(),
        self.values(),
        dst_transposed_sizes,
        self.values().scalar_type(),
        kSparseCsr,
        self.values().device());
    auto dst_transposed = at::zeros(dst_transposed_sizes, self.options().layout(kStrided));
    dst_transposed.add_(to_transposed_csr);
    return dst_transposed.transpose(batch_ndim, batch_ndim + 1);
  }
  if (self.layout() == kSparseBsr || self.layout() == kSparseBsc) {
    Tensor compressed_indices;
    Tensor plain_indices;
    std::tie(compressed_indices, plain_indices) =
        sparse_csr::getCompressedPlainIndices(self);

    auto values = self.values();
    Tensor dense = at::zeros(self.sizes(), self.options().layout(kStrided));
    if (self.dim() == 2) {
      // Pad shape so we can treat 2-d like batched, we will squeeze out the
      // phantom batch dim at the end
      compressed_indices.unsqueeze_(0);
      plain_indices.unsqueeze_(0);
      values = values.unsqueeze_(0);
      dense = dense.unsqueeze_(0);
    }
    if (self.dim() > 3) {
      // Flatten batch dims
      compressed_indices = compressed_indices.flatten(0, batch_ndim - 1);
      plain_indices = plain_indices.flatten(0, batch_ndim - 1);
      values = values.flatten(0, batch_ndim - 1);
      dense = dense.flatten(0, batch_ndim - 1);
    }

    // At this point everything has 3d shape either the batch dim was inserted,
    // existed already or was flattened from multiple batch dims
    std::array<int64_t, 2> blocksize = {values.size(-2), values.size(-1)};
    auto n_batch = values.size(0);
    // If we already had batch dim(s) and any of them were zero we can take the
    // early exit.
    if (n_batch == 0) {
      return dense.reshape(self.sizes());
    }
    // Due to early exit above this reshape should always be valid
    dense = dense.reshape({n_batch, -1, values.size(-2), values.size(-1)});
    for (auto batch : c10::irange(n_batch)) {
      Tensor batch_indices = at::_convert_indices_from_csr_to_coo(
          compressed_indices[batch],
          plain_indices[batch],
          false,
          self.layout() == kSparseBsc);
      auto batch_row_indices = batch_indices.select(0, 0);
      auto batch_col_indices = batch_indices.select(0, 1);
      auto offsets = batch_col_indices +
          batch_row_indices * (self.size(-1) / blocksize[1]);
      dense[batch].index_add_(0, offsets, values[batch]);
    }

    // un-tile the result, NOTE: The final reshape uses the original
    // self.sizes() which will squeeze out the extra batch dim if we put one in
    return dense
        .unflatten(
            1, {self.size(-2) / blocksize[0], self.size(-1) / blocksize[1]})
        .transpose(2, 3)
        .reshape(self.sizes());
  }
  return self.to_sparse().to_dense();
}

// Computes the strides for view_dtype output when the view dtype is
// smaller than the original dtype
inline DimVector compute_strides_for_view_dtype_downsize(IntArrayRef old_strides, int64_t size_ratio, ScalarType old_dtype, ScalarType new_dtype) {
  const int64_t ndim = old_strides.size();

  TORCH_CHECK(
    old_strides[ndim - 1] == 1,
    "self.stride(-1) must be 1 to view ", old_dtype, " as ", new_dtype,
    " (different element sizes), but got ", old_strides[ndim - 1]);

  DimVector new_strides(ndim);
  for (int64_t dim_idx = 0; dim_idx < ndim - 1; dim_idx++) {
    new_strides[dim_idx] = old_strides[dim_idx] * size_ratio;
  }
  new_strides[ndim - 1] = 1;
  return new_strides;
}

// Computes the strides for view_dtype output when the view dtype is
// larger than the original dtype
inline DimVector compute_strides_for_view_dtype_upsize(IntArrayRef old_strides, int64_t size_ratio, ScalarType old_dtype, ScalarType new_dtype) {
  const int64_t ndim = old_strides.size();
  TORCH_CHECK(
    old_strides[ndim - 1] == 1,
    "self.stride(-1) must be 1 to view ", old_dtype, " as ", new_dtype,
    " (different element sizes), but got ", old_strides[ndim - 1]);

  DimVector new_strides(ndim);
  for (int64_t dim_idx = 0; dim_idx < ndim - 1; dim_idx++) {
    TORCH_CHECK(
      (old_strides[dim_idx] % size_ratio) == 0,
      "self.stride(", dim_idx, ") must be divisible by ", size_ratio,
      " to view ", old_dtype, " as ", new_dtype, " (different element sizes), ",
      "but got ", old_strides[dim_idx]);

    new_strides[dim_idx] = old_strides[dim_idx] / size_ratio;
  }
  new_strides[ndim - 1] = 1;
  return new_strides;
}

Tensor view_dtype(const Tensor& self, ScalarType dtype) {
  if (self.scalar_type() == dtype) {
    return self;
  }
  const auto type_meta = c10::scalarTypeToTypeMeta(dtype);
  TORCH_CHECK(!self.is_conj(),
    "torch.Tensor.view is not supported for conjugate view tensors when converting to a different dtype.");
  TORCH_CHECK(!self.is_neg(),
    "torch.Tensor.view is not supported for tensors with negative bit set when converting to a different dtype.");

  int64_t self_element_size = self.element_size();
  int64_t new_element_size = static_cast<int64_t>(type_meta.itemsize());

  Storage storage = self.storage();
  auto new_tensor = detail::make_tensor<TensorImpl>(
      std::move(storage), self.key_set(), type_meta);
  auto* impl = new_tensor.unsafeGetTensorImpl();

  if (self_element_size == new_element_size) {
    impl->set_storage_offset(self.storage_offset());
    impl->set_sizes_and_strides(self.sizes(), self.strides());

  } else if (self.dim() == 0) {
    TORCH_CHECK(false,
      "self.dim() cannot be 0 to view ", self.scalar_type(), " as ",
      dtype, " (different element sizes)");

  } else if (self_element_size > new_element_size) {
    // Downsizing element size

    int64_t size_ratio = self_element_size / new_element_size;
    auto new_strides = compute_strides_for_view_dtype_downsize(
      self.strides(), size_ratio, self.scalar_type(), dtype);

    auto old_sizes = self.sizes();
    DimVector new_sizes(self.dim());
    std::copy(old_sizes.begin(), old_sizes.end(), new_sizes.begin());
    new_sizes[self.dim() - 1] *= size_ratio;

    auto new_storage_offset = size_ratio * self.storage_offset();

    impl->set_storage_offset(new_storage_offset);
    impl->set_sizes_and_strides(new_sizes, new_strides);

  } else {
    // Upsizing element size

    int64_t size_ratio = new_element_size / self_element_size;

    TORCH_CHECK(
      (self.size(-1) % size_ratio) == 0,
      "self.size(-1) must be divisible by ", size_ratio, " to view ",
      self.scalar_type(), " as ", dtype, " (different element sizes), ",
      "but got ", self.size(-1));

    TORCH_CHECK(
      (self.storage_offset() % size_ratio) == 0,
      "self.storage_offset() must be divisible by ", size_ratio, " to view ",
      self.scalar_type(), " as ", dtype, " (different element sizes), but got ",
      self.storage_offset());

    auto new_strides = compute_strides_for_view_dtype_upsize(
      self.strides(), size_ratio, self.scalar_type(), dtype);

    auto old_sizes = self.sizes();
    DimVector new_sizes(self.dim());
    std::copy(old_sizes.begin(), old_sizes.end(), new_sizes.begin());
    new_sizes[self.dim() - 1] /= size_ratio;

    auto new_storage_offset = self.storage_offset() / size_ratio;

    impl->set_storage_offset(new_storage_offset);
    impl->set_sizes_and_strides(new_sizes, new_strides);
  }

  return new_tensor;
}

Tensor _tile_tensor(const Tensor& self, IntArrayRef blocksize) {
  // This code turns a matrix into a sequence of blocks
  //
  // Given matrix
  //
  //  1  2  3  4
  //  5  6  7  8
  //  9 10 11 12
  // 14 15 16 17
  //
  // _tile_tensor(matrix, {2, 2}) will yield the following 2 by 2 blocks
  //
  //  1  2 |  3  4 |  9 10 | 11 12
  //  5  6 |  7  8 | 14 15 | 16 17
  //
  //  via a 4D Tensor of shape (2, 2, 2, 2)
  //
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(blocksize[0] > 0);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(blocksize[1] > 0);
  auto block_size_0 = self.size(0) / blocksize[0];
  auto block_size_1 = self.size(1) / blocksize[1];
  return self.reshape({block_size_0, blocksize[0], block_size_1, blocksize[1]})
      .transpose(1, 2)
      .contiguous();
}

Tensor _batch_tile_tensor(const Tensor& self, IntArrayRef blocksize) {
  if (self.dim() == 2) {
    return _tile_tensor(self, blocksize);
  }
  auto n_batch_dim = self.dim() - 2;
  // Same as _tile_tensor, just per matrix entry of self, if self is 3D.
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(blocksize[0] > 0);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(blocksize[1] > 0);
  auto block_size_0 = self.size(-2) / blocksize[0];
  auto block_size_1 = self.size(-1) / blocksize[1];
  auto tiled_sizes = DimVector(self.sizes().slice(0, n_batch_dim));
  tiled_sizes.push_back(block_size_0);
  tiled_sizes.push_back(blocksize[0]);
  tiled_sizes.push_back(block_size_1);
  tiled_sizes.push_back(blocksize[1]);

  return self.reshape(tiled_sizes).transpose(-3, -2).contiguous();
}

Tensor _mask_to_indices(const Tensor& mask) {
  // This function returns a vector of the indices at which given
  // boolean mask is True. at::nonzero can achieve the same, but
  // we yet have to compare the performance difference.
  TORCH_CHECK(mask.dim() == 1, "Currently _mask_to_indices only supports 1-d masks.");
  TORCH_CHECK(mask.dtype() == at::kBool, "Expected mask to be of dtype bool.");
  return at::native::arange(
      mask.numel(), at::kLong, kStrided, mask.device())
      .masked_select(mask);
}

std::pair<Tensor, Tensor> _not_zero_mask_to_col_row_indices(
    Tensor not_zero_mask,
    ScalarType index_dtype,
    Device index_device) {
  auto col_indices =
      at::native::arange(not_zero_mask.size(-1), index_dtype, kStrided, index_device)
          .view({1, not_zero_mask.size(-1)})
          .expand_as(not_zero_mask)
          .masked_select(not_zero_mask);
  auto row_indices =
      at::native::arange(
          not_zero_mask.size(-2), index_dtype, kStrided, index_device)
          .view({not_zero_mask.size(-2), 1})
          .expand_as(not_zero_mask)
          .masked_select(not_zero_mask);
  return std::pair<Tensor, Tensor>(col_indices, row_indices);
}

// Sparse layout conversions Start

Tensor dense_to_sparse_csr(const Tensor& self) {
  auto n_batch_dim = self.dim() - 2;
  auto values = self;
  auto not_zero_mask = self != 0;

  if (n_batch_dim > 0) {
    dense_to_sparse_compressed_prepare_check_mask_values_batched(
        Layout::SparseCsr, values, not_zero_mask, n_batch_dim);
  }

  Tensor col_indices;
  Tensor row_indices;
  std::tie(col_indices, row_indices) = _not_zero_mask_to_col_row_indices(
      not_zero_mask, at::kLong, not_zero_mask.device());
  Tensor crow_indices = at::_convert_indices_from_coo_to_csr(
      row_indices, not_zero_mask.size(0), false /*out_int32*/);
  {
    auto mask_indices = _mask_to_indices(not_zero_mask.flatten());
    values = values.flatten().index_select(0, mask_indices);
  }

  if (n_batch_dim > 0) {
    reshape_2d_sparse_compressed_members_to_nd_batched(
        self.sizes(), n_batch_dim, crow_indices, col_indices, values);
  }
  return at::native::_sparse_csr_tensor_unsafe(
      crow_indices,
      col_indices,
      values,
      self.sizes(),
      values.scalar_type(),
      c10::kSparseCsr,
      values.device());
}

Tensor dense_to_sparse_csc(const Tensor& self) {
  auto n_batch_dim = self.dim() - 2;
  auto values = self;
  auto not_zero_mask = self != 0;

  if (n_batch_dim > 0) {
    dense_to_sparse_compressed_prepare_check_mask_values_batched(
        Layout::SparseCsc, values, not_zero_mask, n_batch_dim);
  }

  Tensor col_indices;
  Tensor row_indices;
  // Compressed col indices are the same as the row indices of the transpose!
  std::tie(row_indices, col_indices) = _not_zero_mask_to_col_row_indices(
      not_zero_mask.transpose(1, 0), at::kLong, not_zero_mask.device());
  Tensor ccol_indices = at::_convert_indices_from_coo_to_csr(
      col_indices, not_zero_mask.size(-1), false /*out_int32*/);
  {
    // We need to transpose the mask and values before flattening so the nnz dim
    // will run in col-major order.
    values = values.transpose(0, 1).flatten();
    auto mask_indices =
        _mask_to_indices(not_zero_mask.transpose(0, 1).flatten());
    values = values.index_select(0, mask_indices);
  }

  if (n_batch_dim > 0) {
    reshape_2d_sparse_compressed_members_to_nd_batched(
        self.sizes(), n_batch_dim, ccol_indices, row_indices, values);
  }
  return at::native::_sparse_csc_tensor_unsafe(
      ccol_indices,
      row_indices,
      values,
      self.sizes(),
      values.scalar_type(),
      c10::kSparseCsc,
      values.device());
}

Tensor dense_to_sparse_bsr(const Tensor& self, IntArrayRef blocksize) {
  TORCH_CHECK(
      blocksize[0] > 0 && blocksize[1] > 0,
      "blocksize needs to be non zero, but got ",
      blocksize);
  TORCH_CHECK(
      self.size(-2) % blocksize[0] == 0,
      "Tensor size(-2) ",
      self.size(-2),
      " needs to be divisible by blocksize[0] ",
      blocksize[0]);
  TORCH_CHECK(
      self.size(-1) % blocksize[1] == 0,
      "Tensor size(-1) ",
      self.size(-1),
      " needs to be divisible by blocksize[1] ",
      blocksize[1]);

  auto n_batch_dim = self.dim() - 2;

  auto values = _batch_tile_tensor(self, blocksize);
  auto not_zero_mask = _batch_tile_tensor((self != 0), blocksize);
  auto mask_shape = DimVector(not_zero_mask.sizes().slice(0, n_batch_dim + 2));
  // Can't use -1 here one of sparse/batch dims may be zero
  mask_shape.push_back(blocksize[0] * blocksize[1]);
  not_zero_mask = not_zero_mask.view(mask_shape).any(-1);

  if (n_batch_dim > 0) {
    dense_to_sparse_compressed_prepare_check_mask_values_batched(
        Layout::SparseBsr, values, not_zero_mask, n_batch_dim);
  }

  Tensor col_indices;
  Tensor row_indices;
  std::tie(col_indices, row_indices) = _not_zero_mask_to_col_row_indices(
      not_zero_mask, at::kLong, not_zero_mask.device());

  Tensor crow_indices = at::_convert_indices_from_coo_to_csr(
      row_indices, not_zero_mask.size(0), false /*out_int32*/);

  {
    auto mask_indices = _mask_to_indices(not_zero_mask.flatten());
    values = values.flatten(0, -3).index_select(0, mask_indices);
  }

  if (n_batch_dim > 0) {
    reshape_2d_sparse_compressed_members_to_nd_batched(
        self.sizes(), n_batch_dim, crow_indices, col_indices, values);
  }
  return at::native::_sparse_bsr_tensor_unsafe(
      crow_indices,
      col_indices,
      values,
      self.sizes(),
      values.scalar_type(),
      c10::kSparseBsr,
      values.device());
}

Tensor dense_to_sparse_bsc(const Tensor& self, IntArrayRef blocksize) {
  TORCH_CHECK(
      blocksize[0] > 0 && blocksize[1] > 0,
      "blocksize needs to be non zero, but got ",
      blocksize);
  TORCH_CHECK(
      self.size(-2) % blocksize[0] == 0,
      "Tensor size(-2) ",
      self.size(-2),
      " needs to be divisible by blocksize[0] ",
      blocksize[0]);
  TORCH_CHECK(
      self.size(-1) % blocksize[1] == 0,
      "Tensor size(-1) ",
      self.size(-1),
      " needs to be divisible by blocksize[1] ",
      blocksize[1]);
  auto n_batch_dim = self.dim() - 2;
  auto is_batched = n_batch_dim > 0;
  auto values = _batch_tile_tensor(self, blocksize);
  auto not_zero_mask = _batch_tile_tensor((self != 0), blocksize);
  auto mask_shape = DimVector(not_zero_mask.sizes().slice(0, n_batch_dim + 2));
  // Can't use -1 here one of sparse/batch dims may be zero
  mask_shape.push_back(blocksize[0] * blocksize[1]);
  not_zero_mask = not_zero_mask.view(mask_shape).any(-1);

  if (is_batched) {
    dense_to_sparse_compressed_prepare_check_mask_values_batched(
        Layout::SparseBsc, values, not_zero_mask, n_batch_dim);
  }

  Tensor col_indices;
  Tensor row_indices;
  // Compressed col indices are the same as the row indices of the transpose!
  std::tie(row_indices, col_indices) = _not_zero_mask_to_col_row_indices(
      not_zero_mask.transpose(1, 0), at::kLong, not_zero_mask.device());
  // This only works if the col_indices vector is in ascending order.
  Tensor ccol_indices = at::_convert_indices_from_coo_to_csr(
      col_indices, not_zero_mask.size(-1), false /*out_int32*/);
  {
    // We need the block-values in col major order, but blocks themselves to
    // remain in row-major order, so we transpose the leading two dims, leaving
    // the trailing two dims as is.
    values = values.transpose(0, 1).flatten(0, -3);
    // The mask must transpose as well to index it correctly.
    auto mask_indices =
        _mask_to_indices(not_zero_mask.transpose(0, 1).flatten());
    values = values.index_select(0, mask_indices);
  }
  if (is_batched) {
    reshape_2d_sparse_compressed_members_to_nd_batched(
        self.sizes(), n_batch_dim, ccol_indices, row_indices, values);
  }

  return at::native::_sparse_bsc_tensor_unsafe(
      ccol_indices,
      row_indices,
      values,
      self.sizes(),
      values.scalar_type(),
      c10::kSparseBsc,
      values.device());
}

void _check_blocksize_matches(
    const Tensor& self,
    c10::optional<IntArrayRef> blocksize_opt,
    const std::string& name) {
  if (blocksize_opt.has_value()) {
    const auto blocksize = *blocksize_opt;
    const auto self_values = self.values();
    const auto self_blocksize = at::DimVector({self_values.size(-2), self_values.size(-1)});
    TORCH_CHECK(self_blocksize == blocksize,
        name, "(): the provided blocksize does not match the blocksize of the to be converted tensor, ",
        "got (", blocksize[0], ", ", blocksize[1], ") ",
        "but expected (", self_blocksize[0], ", ", self_blocksize[1], ").");
  }
}

Tensor sparse_compressed_clone(
    const Tensor& self,
    c10::optional<IntArrayRef> blocksize,
    const std::string& name) {
  _check_blocksize_matches(self, blocksize, name);
  // Just returning self doesn't work
  // RuntimeError: t.use_count() <= 1 INTERNAL ASSERT FAILED at
  // "../torch/csrc/autograd/autograd_not_implemented_fallback.cpp":152,
  // please report a bug to PyTorch.
  const auto layout = self.layout();
  Tensor compressed_indices, plain_indices;
  std::tie(compressed_indices, plain_indices) = at::sparse_csr::getCompressedPlainIndices(self);
  auto values = self.values();
  return _sparse_compressed_tensor_unsafe(
      compressed_indices,
      plain_indices,
      values,
      self.sizes(),
      values.scalar_type(),
      layout,
      values.device());
}

Tensor sparse_compressed_to_flipped(
    const Tensor& self,
    c10::optional<IntArrayRef> blocksize,
    const std::string& name) {
  _check_blocksize_matches(self, blocksize, name);

  const auto layout = self.layout();
  // NOTE: errors on non-compressed sparse layouts.
  const auto flipped_layout = at::sparse_csr::flip_compressed_layout(layout);

  // Suppose compressed_indices represent rows of an input in either
  // CSR or BSR sparse compressed format.
  // In order to convert a batched CSR/BSR index into a batched CSC/BSC index
  // we perform the following steps:
  // 1. Convert a sparse compressed index representing batches of matrices of
  //    shape (b, r, c) to a sparse compressed index that represents a single
  //    matrix of shape (b * r, c).
  // 2. Turn the compressed indices of the matrix of shape (b * r, c) into
  //    COO indices.
  // 3. Map these COO indices into the COO indices of a matrix of shape (r, b * c)
  //    such that if A is a matrix of shape (b * r, c) and B is a matrix of shape
  //    (r, b * c) such that
  //    A[(k * r):(k * r + r), :] = B[:, (k * c):(k * c + c)] for all k in arange(b),
  //    then A[i, j] = B[i', j'].
  //    This is equivalent to finding indices that match values of matrices
  //    tiled vertically to values of the same matrices tiled horizontally.
  // 4. Convert the COO indices to the CSC/BSC indices and form the output.
  //
  // NOTE: the reason behind vertical/horizontal tiling is to be able to transform
  //       indices over all matrices in the batch in a single kernel call, since
  //       all the existing coo <-> compressed indices conversion methods assume
  //       a single matrix.
  //
  // CSC/BSC inputs are handled in a similar fashion with a "transposed" argument.
  // See the comments below for detailed explanations on how exactly each step
  // is performed.

  Tensor compressed_indices, plain_indices;
  std::tie(compressed_indices, plain_indices) = at::sparse_csr::getCompressedPlainIndices(self);
  auto values = self.values();
  const auto nnz = plain_indices.size(-1);

  const auto n_batches = compressed_indices.dim() - 1;
  auto n_batches_nonzero = n_batches;
  // Insert fake batch dim for simplicity
  if (!n_batches) {
    n_batches_nonzero = 1;
    compressed_indices.unsqueeze_(0);
    plain_indices.unsqueeze_(0);
    values.unsqueeze_(0);
  }

  // NOTE: these sparse_dims are true sparse dims only for CSR/CSC inputs.
  // And for BSR/BSC these are <true sparse dims> / <blocksize>.
  // In other words, sparse_dims stores ranges of valid indices in the row/col dims.
  const auto sparse_dims = [&]() -> at::DimVector {
    auto sparse_dims = at::DimVector(self.sizes().slice(n_batches, 2));
    if (layout == at::kSparseBsr || layout == at::kSparseBsc) {
      std::array<int64_t, 2> blocksize = {values.size(-2), values.size(-1)};
      sparse_dims[0] /= blocksize[0];
      sparse_dims[1] /= blocksize[1];
    }
    return sparse_dims;
  }();

  // batch_sizes_nonempty stores at least one, potentially fake, batch dimension.
  // rebatch_sizes_nonempty is equivalent to batch_sizes_nonempty.push_back(-1),
  // and is used to unflatten batch dimensions from a dimension of size
  // (batch_numel * dim_size,) for some dim_size.
  const auto batch_sizes_nonempty = at::DimVector(plain_indices.sizes().slice(0, n_batches_nonzero));
  auto rebatch_sizes_nonempty = at::DimVector(batch_sizes_nonempty);
  rebatch_sizes_nonempty.push_back(-1);
  const auto batch_numel_nonzero = std::accumulate(
      batch_sizes_nonempty.begin(),
      batch_sizes_nonempty.begin() + n_batches_nonzero,
      1,
      std::multiplies<int64_t>());

  // Equivalent to (arange(batch_numel_nonzero).mul_(nnz)).reshape(batch_sizes_nonempty).
  // We just compute it differently to use `add` kernel in place of `mul` for better
  // performance.
  const auto batch_nnz_offset = [&]() -> Tensor {
    const auto wrapped_nnz = at::tensor({nnz}, compressed_indices.options());
    const auto offset = wrapped_nnz
      .expand({batch_numel_nonzero})
      .cumsum(-1).sub_(wrapped_nnz)
      .reshape(batch_sizes_nonempty);
    return offset;
  }();

  // Step 1 for CSR/BSR inputs:
  // Convert a sparse compressed index representing batches of matrices of
  // shape (b, r, c) to a sparse compressed index that represents a single
  // matrix of shape (b * r, c).
  // The algorithm is identical for CSC/BSC inputs, with the batch dimensions
  // flattened in the "transposed" dimension.
  const auto compressed_indices_2d = [&]() -> Tensor {
    // Extract offsets only relevant for the first :-1 elements in a row/col.
    const auto compressed_offsets = compressed_indices.slice(-1, 0, -1);
    // batch_offsets offsets each individual matrix row/col offsets by the total
    // sum of nnz's of all the matrices with the smaller batch index.
    const auto batch_offsets = batch_nnz_offset
      .unsqueeze(-1).expand_as(compressed_offsets);
    // compressed_offsets + batch_offsets creates an offset vector for a 2d matrix
    // that is stored in a compressed sparse format.
    const auto compressed_offsets_2d = compressed_offsets.add(batch_offsets).reshape({-1});
    const auto offsets_len = compressed_offsets_2d.numel();
    auto res = at::empty({offsets_len + 1}, compressed_indices.options());
    res.slice(-1, 0, -1).copy_(compressed_offsets_2d);
    // By appending nnz * batch_numel_nonzero to (compressed_offsets + batch_offsets)
    // a compressed index of a 2d matrix is formed.
    res.slice(-1, -1).fill_(nnz * batch_numel_nonzero);
    return res;
  }();
  // More involved for compressed indices, but pretty easy for plain_indices and values:
  // just squash batch dimensions.
  const auto plain_indices_2d = plain_indices.flatten(0, n_batches_nonzero);
  // NOTE: values are not 2d! They just represent values of a sparse compressed 2d matrix.
  const auto values_2d = values.flatten(0, n_batches_nonzero);

  const auto is_out_int32 = compressed_indices.scalar_type() == ScalarType::Int;

  // Step 2 & 3:
  //
  // Turn the compressed indices of the matrix of shape (b * r, c) into COO indices.
  //
  // Map these COO indices into the COO indices of a matrix of shape (r, b * c)
  // such that if A is a matrix of shape (b * r, c) and B is a matrix of shape
  // (r, b * c) such that
  // A[(k * r):(k * r + r), :] = B[:, (k * c):(k * c + c)] for all k in arange(b),
  // then A[i, j] = B[i', j'].
  // This is equivalent to finding indices that match values of matrices
  // tiled vertically to values of the same matrices tiled horizontally.

  // coo <-> sparse index conversions assume CSR/BSR inputs.
  // To CSC/BSC inputs these indices will appear "transposed".
  const auto is_transposed_indices = layout == at::kSparseCsc || layout == at::kSparseBsc;
  const auto coo_indices_2d_transposed = [&]() -> Tensor {
    const auto coo_indices_2d = _convert_indices_from_csr_to_coo(
        compressed_indices_2d,
        plain_indices_2d,
        is_out_int32,
        /*transpose=*/true); // Flip rows/cols for convenience.
    // Convert COO indices of (b * r, c) to (r, b * c).
    // It is a map (i, j) -> {
    //    b = i // r
    //    i' = i % r
    //    j' = j + b * c
    //    return (i', j')
    // }
    // NOTE: we used transposed=true above!
    auto i = coo_indices_2d.select(0, 1);
    auto j = coo_indices_2d.select(0, 0);
    auto b = i.div(is_transposed_indices ? sparse_dims[1] : sparse_dims[0], "trunc");
    // Modify i, j in-place.
    i.fmod_(is_transposed_indices ? sparse_dims[1] : sparse_dims[0]);
    j.add_(b * (is_transposed_indices ? sparse_dims[0] : sparse_dims[1]));
    return coo_indices_2d;
  }();

  // Step 4:
  // Convert the COO indices to the CSC/BSC indices and form the output.
  // We need to sort COO indices along the "tranposed" dim to satisfy the
  // invariant of sorted plain indices.
  // Hash coo indices by converting 2d indices to linear offsets with
  // more "weight" (aka stride) placed on the "transposed" dimension.
  const auto coo_indices_2d_transposed_hashed = at::sparse::flatten_indices(
      coo_indices_2d_transposed,
      is_transposed_indices ? at::DimVector({sparse_dims[0], sparse_dims[1] * batch_numel_nonzero})
                            : at::DimVector({sparse_dims[1], sparse_dims[0] * batch_numel_nonzero}));
  const auto hash_argsort = std::get<1>(coo_indices_2d_transposed_hashed.sort());
  const auto coo_indices_2d_transposed_sorted = coo_indices_2d_transposed.index_select(1, hash_argsort);

  const auto new_compressed_indices_coo_2d = coo_indices_2d_transposed_sorted.select(0, 0);
  const auto new_plain_indices_2d = coo_indices_2d_transposed_sorted.select(0, 1);
  const auto new_values_2d = values_2d.index_select(0, hash_argsort);

  auto new_compressed_indices = compressed_to_batched_compressed_indices(
      _convert_indices_from_coo_to_csr(
        new_compressed_indices_coo_2d,
        is_transposed_indices
          ? batch_numel_nonzero * sparse_dims[0]
          : batch_numel_nonzero * sparse_dims[1],
        is_out_int32),
      batch_numel_nonzero,
      is_out_int32)
    .unflatten(0, batch_sizes_nonempty);
  auto new_plain_indices = new_plain_indices_2d.unflatten(0, rebatch_sizes_nonempty);
  auto new_values = new_values_2d.unflatten(0, rebatch_sizes_nonempty);
  // Kill fake batch dim if it was inserted.
  if (!n_batches) {
    new_compressed_indices.squeeze_(0);
    new_plain_indices.squeeze_(0);
    new_values.squeeze_(0);
  }

  return _sparse_compressed_tensor_unsafe(
      new_compressed_indices,
      new_plain_indices,
      new_values,
      self.sizes(),
      new_values.scalar_type(),
      flipped_layout,
      new_values.device());
}

Tensor sparse_compressed_to_sparse_csr(const Tensor& self) {
  if (self.layout() == kSparseCsc) {
    return sparse_compressed_to_flipped(self, c10::nullopt, "to_sparse_csr");
  }
  if (self.layout() == kSparseCsr) {
    return sparse_compressed_clone(self, c10::nullopt, "to_sparse_csr");
  }
  AT_ERROR(
      "sparse_compressed_to_sparse_csr expected SparseCsr or SparseCsc layout but got ",
      self.layout());
}

Tensor coo_to_sparse_csr(const Tensor& self) {
  TORCH_CHECK(
      self.dim() == 2,
      "Only 2D tensors can be converted to the SparseCsr layout but got shape: ",
      self.sizes());
  auto coalesced_self = self.coalesce();
  auto row_indices = coalesced_self.indices()[0];
  bool out_int32 = (row_indices.scalar_type() == at::kInt);
  auto crow_indices = at::_convert_indices_from_coo_to_csr(
      row_indices, self.size(0), out_int32);
  return at::native::_sparse_csr_tensor_unsafe(
      crow_indices,
      coalesced_self.indices()[1].contiguous(),
      coalesced_self.values(),
      coalesced_self.sizes(),
      coalesced_self.scalar_type(),
      c10::kSparseCsr,
      coalesced_self.device());
}

Tensor coo_to_sparse_csc(const Tensor& self) {
  TORCH_CHECK(
      self.dim() == 2,
      "Only 2D tensors can be converted to the SparseCsc layout but got shape: ",
      self.sizes());
  auto coalesced_self = self.transpose(0, 1).coalesce().to_sparse_csr();
  return at::native::_sparse_csc_tensor_unsafe(
      coalesced_self.crow_indices(),
      coalesced_self.col_indices(),
      coalesced_self.values(),
      self.sizes(),
      coalesced_self.scalar_type(),
      c10::kSparseCsc,
      coalesced_self.device());
}

Tensor coo_to_sparse_bsr(const Tensor& self, IntArrayRef blocksize) {
  AT_ERROR(
      "Conversion from ", self.layout(), " to SparseBsr is currently not supported.");
  return self;
}

Tensor coo_to_sparse_bsc(const Tensor& self, IntArrayRef blocksize) {
  AT_ERROR(
      "Conversion from ", self.layout(), " to SparseBsc is currently not supported.");
  return self;
}

namespace {
template <typename input_t, typename output_t>
void convert_indices_from_coo_to_csr_cpu(
    const Tensor& result,
    const Tensor& input,
    const int64_t size) {
  int64_t numel = input.numel();
  const input_t* data_in = input.data_ptr<input_t>();
  output_t* data_out = result.data_ptr<output_t>();

  if (numel == 0) {
    result.zero_();
    return;
  }

  for (int64_t i = 0; i <= data_in[0]; i++)
    data_out[i] = static_cast<output_t>(0);

  at::parallel_for(
      0, numel - 1, at::internal::GRAIN_SIZE, [&](int64_t start, int64_t end) {
        input_t curr_value = data_in[start], next_value;
        for (const auto i : c10::irange(start, end)) {
          next_value = data_in[i + 1];
          for (; curr_value < next_value; curr_value++)
            data_out[curr_value + 1] = static_cast<output_t>(i + 1);
        }
      });
  for (int64_t i = data_in[numel - 1] + 1; i < size + 1; i++) {
    data_out[i] = static_cast<output_t>(numel);
  }
}

template <typename input_t, typename output_t>
void convert_indices_from_csr_to_coo_cpu(
    const Tensor& indices,
    const Tensor& crow_indices,
    const Tensor& col_indices,
    const bool transpose = false) {
  int64_t nrows = crow_indices.numel() - 1;
  if (nrows == 0) {
    indices.zero_();
    return;
  }
  auto crow_indices_ = crow_indices.expect_contiguous();
  const input_t* crow_indices_data_in = crow_indices_->data_ptr<input_t>();
  TORCH_INTERNAL_ASSERT(indices.is_contiguous());
  auto row0 = indices.select(0, transpose ? 1 : 0);
  auto row1 = indices.select(0, transpose ? 0 : 1);
  output_t* data_out = row0.data_ptr<output_t>();
  row1.copy_(*col_indices.expect_contiguous());
  at::parallel_for(
      0, nrows, at::internal::GRAIN_SIZE, [&](int64_t start, int64_t end) {
        for (const auto i : c10::irange(start, end)) {
          std::fill(
              &data_out[crow_indices_data_in[i]],
              &data_out[crow_indices_data_in[i + 1]],
              static_cast<output_t>(i));
        }
      });
}
} // namespace

TORCH_IMPL_FUNC(_convert_indices_from_coo_to_csr_structured_cpu)
(const Tensor& input,
 const int64_t size,
 const bool out_int32,
 const Tensor& result) {
  if (out_int32) {
    AT_DISPATCH_INTEGRAL_TYPES(
        input.scalar_type(), "convert_indices_from_coo_to_csr_cpu", [&] {
          convert_indices_from_coo_to_csr_cpu<scalar_t, int32_t>(
              result, input, size);
        });
  } else {
    AT_DISPATCH_INTEGRAL_TYPES(
        input.scalar_type(), "convert_indices_from_coo_to_csr_cpu", [&] {
          convert_indices_from_coo_to_csr_cpu<scalar_t, int64_t>(
              result, input, size);
        });
  }
}

TORCH_IMPL_FUNC(_convert_indices_from_csr_to_coo_structured_cpu)
(const Tensor& crow_indices,
 const Tensor& col_indices,
 const bool out_int32,
 const bool transpose,
 const Tensor& result) {
  if (out_int32) {
    AT_DISPATCH_INTEGRAL_TYPES(
        crow_indices.scalar_type(), "convert_indices_from_csr_to_coo_cpu", [&] {
          convert_indices_from_csr_to_coo_cpu<scalar_t, int32_t>(
              result, crow_indices, col_indices, transpose);
        });
  } else {
    AT_DISPATCH_INTEGRAL_TYPES(
        crow_indices.scalar_type(), "convert_indices_from_csr_to_coo_cpu", [&] {
          convert_indices_from_csr_to_coo_cpu<scalar_t, int64_t>(
              result, crow_indices, col_indices, transpose);
        });
  }
}

/*
 * Based on
 * https://github.com/scipy/scipy/blob/8a64c938ddf1ae4c02a08d2c5e38daeb8d061d38/scipy/sparse/sparsetools/csr.h
 */
template <class I, class T>
void _csr_to_block_csr_cpu_kernel(
    const I n_row,
    const I n_col,
    const I R,
    const I C,
    const I* input_crow_indices,
    const I* input_col_indices,
    const T* input_values,
    I* result_crow_indices,
    I* result_col_indices,
    T* result_values) {
  // All blocks are possible, that is, may be allocated if a single non-zero
  // value lives within them. Otherwise they're not.

  // Allocate pointers for all possible column blocks plus 1
  std::vector<T*> blocks(n_col / C + 1, (T*)0);

  assert(n_row % R == 0);
  assert(n_col % C == 0);

  // Major assumptions
  // 1. Blocks must be square

  // Number of blocks along rows
  I n_brow = n_row / R;
  // Number of blocks along columns
  // I n_bcol = n_col / C;

  // Number of elements per block
  I RC = R * C;
  // Number of blocks overall
  I n_blks = 0;

  result_crow_indices[0] = 0;

  // Iterate over blocks along rows
  for (I block_i = 0; block_i < n_brow; block_i++) {
    // Iterate over rows within block
    for (I r = 0; r < R; r++) {
      I i = R * block_i + r; // row index
      for (I jj = input_crow_indices[i]; jj < input_crow_indices[i + 1]; jj++) {
        I j = input_col_indices[jj]; // column index

        // Block corresponding to column index
        I block_j = j / C;
        // Column within block
        I c = j % C;

        if (blocks[block_j] == 0) {
          blocks[block_j] = result_values + RC * n_blks;
          result_col_indices[n_blks] = block_j;
          n_blks++;
        }

        // Specific blocks entries should not be visited more than once.
        // Scipy code does an addition here. Why?
        *(blocks[block_j] + C * r + c) = input_values[jj];
      }
    }

    for (I jj = input_crow_indices[R * block_i];
         jj < input_crow_indices[R * (block_i + 1)];
         jj++) {
      blocks[input_col_indices[jj] / C] = 0;
    }

    result_crow_indices[block_i + 1] = n_blks;
  }
}

/*
 * Based on
 * https://github.com/scipy/scipy/blob/8a64c938ddf1ae4c02a08d2c5e38daeb8d061d38/scipy/sparse/sparsetools/csr.h
 */
template <class I>
I csr_count_blocks(
    const I n_row,
    const I n_col,
    const I R,
    const I C,
    const I Ap[],
    const I Aj[]) {
  std::vector<I> mask(n_col / C + 1, -1);
  I n_blks = 0;
  for (I i = 0; i < n_row; i++) {
    I bi = i / R;
    for (I jj = Ap[i]; jj < Ap[i + 1]; jj++) {
      I bj = Aj[jj] / C;
      if (mask[bj] != bi) {
        mask[bj] = bi;
        n_blks++;
      }
    }
  }
  return n_blks;
}

Tensor _csr_to_block_csr_cpu(const Tensor& self, IntArrayRef blocksize) {
  TORCH_CHECK(
      blocksize[0] == blocksize[1],
      "blocks must be square. ",
      "Got (",
      blocksize[0],
      ", ",
      blocksize[1],
      ") instead.");
  TORCH_CHECK(
      self.size(0) % blocksize[0] == 0 && self.size(1) % blocksize[1] == 0,
      "Block sparse CSR Tensors must have a size that is an ",
      "integral multiple of their block size. ",
      "Got Tensor of size (",
      self.size(0),
      ", ",
      self.size(1),
      ") with block size (",
      blocksize[0],
      ", ",
      blocksize[1],
      ") instead.");
  Tensor input_values = self.values().contiguous();
  Tensor input_crow_indices = self.crow_indices().contiguous();
  Tensor input_col_indices = self.col_indices().contiguous();

  // First we determine the number of blocks needed. For each given block, if it
  // contains a non-zero element we will allocate values and indices for it.
  int64_t num_blocks;
  int64_t n_row = self.size(0);
  int64_t n_col = self.size(1);
  AT_DISPATCH_INDEX_TYPES(
      input_crow_indices.scalar_type(), "_csr_to_block_csr_cpu", [&] {
        num_blocks = csr_count_blocks<index_t>(
            n_row,
            n_col,
            blocksize[0],
            blocksize[1],
            input_crow_indices.data_ptr<index_t>(),
            input_col_indices.data_ptr<index_t>());
      });

  Tensor result_values =
      input_values.new_zeros({num_blocks, blocksize[0], blocksize[1]});
  Tensor result_crow_indices =
      input_crow_indices.new_empty({(n_row / blocksize[0]) + 1});
  Tensor result_col_indices = input_col_indices.new_empty({num_blocks});

  // Next we copy over non-zero elements into the allocated blocks.
  AT_DISPATCH_INDEX_TYPES(
      input_crow_indices.scalar_type(), "_csr_to_block_csr_cpu", [&] {
        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
            input_values.scalar_type(), "_csr_to_block_csr_cpu", [&] {
              _csr_to_block_csr_cpu_kernel<index_t, scalar_t>(
                  n_row,
                  n_col,
                  blocksize[0],
                  blocksize[1],
                  input_crow_indices.data_ptr<index_t>(),
                  input_col_indices.data_ptr<index_t>(),
                  input_values.data_ptr<scalar_t>(),
                  result_crow_indices.data_ptr<index_t>(),
                  result_col_indices.data_ptr<index_t>(),
                  result_values.data_ptr<scalar_t>());
            });
      });
  return at::native::_sparse_bsr_tensor_unsafe(
      result_crow_indices,
      result_col_indices,
      result_values,
      self.sizes(),
      result_values.scalar_type(),
      c10::kSparseBsr,
      result_values.device());
}

Tensor sparse_compressed_to_sparse_bsr(const Tensor& self, IntArrayRef blocksize) {
  if (self.layout() == kSparseBsc) {
    return sparse_compressed_to_flipped(self, blocksize, "to_sparse_bsr");
  }
  if (self.layout() == kSparseBsr) {
    return sparse_compressed_clone(self, blocksize, "to_sparse_bsr");
  }
  if (self.layout() == kSparseCsr) {
    TORCH_CHECK(self.dim() == 2,
        "to_sparse_bsr(): conversion from Csr to Bsr is only possible for 2d inputs, ",
        "but got input of dimension ", self.dim(), " instead.");
    Tensor self_values = self.values();
    Tensor self_crow_indices = self.crow_indices();
    Tensor self_col_indices = self.col_indices();
    Tensor cpu_result = _csr_to_block_csr_cpu(
        _sparse_csr_tensor_unsafe(
            self_crow_indices.cpu(),
            self_col_indices.cpu(),
            self_values.cpu(),
            self.sizes(),
            self_values.scalar_type(),
            self.layout(),
            at::kCPU),
        blocksize);
    Tensor result_values = cpu_result.values().to(self_values.options());
    Tensor result_crow_indices =
        cpu_result.crow_indices().to(self_crow_indices.options());
    Tensor result_col_indices =
        cpu_result.col_indices().to(self_col_indices.options());
    return at::native::_sparse_bsr_tensor_unsafe(
        result_crow_indices,
        result_col_indices,
        result_values,
        self.sizes(),
        result_values.scalar_type(),
        c10::kSparseBsr,
        result_values.device());
  }
  AT_ERROR(
      "sparse_compressed_to_sparse_bsr expected SparseCsr, SparseBsr or SparseBsc layout but got ",
      self.layout());
  return self;
}

Tensor sparse_compressed_to_sparse_bsc(const Tensor& self, IntArrayRef blocksize) {
  if (self.layout() == kSparseBsr) {
    return sparse_compressed_to_flipped(self, blocksize, "to_sparse_bsr");
  }
  if (self.layout() == kSparseBsc) {
    return sparse_compressed_clone(self, blocksize, "to_sparse_bsr");
  }
  AT_ERROR(
      "sparse_compressed_to_sparse_bsc expected SparseBsr or SparseBsc layout but got ",
      self.layout());
  return self;
}

Tensor sparse_compressed_to_sparse_csc(const Tensor& self) {
  if (self.layout() == kSparseCsr) {
    return sparse_compressed_to_flipped(self, c10::nullopt, "to_sparse_csc");
  }
  if (self.layout() == kSparseCsc) {
    return sparse_compressed_clone(self, c10::nullopt, "to_sparse_csc");
  }
  AT_ERROR(
      "sparse_compressed_to_sparse_csc expected SparseCsr or SparseCsc layout but got ",
      self.layout());
  return self;
}

Tensor sparse_compressed_to_sparse(const Tensor& self, int64_t sparse_dim) {
  TORCH_CHECK(sparse_dim > 0, "sparse_dim must be >0");
  TORCH_CHECK(sparse_dim <= 2,
              "sparse_dim must be less than or equal to 2");
  // TODO: implement coo.to_sparse(sparse_dim) and then use
  // return self.to_sparse().to_sparse(sparse_dim);
  TORCH_CHECK(
      sparse_dim == 2, "sparse dim 1 is not supported by sparse_csr_to_dense");
  if (self.layout() == kSparseCsc) {
    Tensor indices = at::_convert_indices_from_csr_to_coo(
        self.ccol_indices(), self.row_indices(), false, true);
    return at::native::_sparse_coo_tensor_unsafe(
               indices, self.values(), self.sizes())
        ._coalesced_(true);
  }
  if (self.layout() == kSparseCsr) {
    Tensor indices = at::_convert_indices_from_csr_to_coo(
        self.crow_indices(), self.col_indices(), false, false);
    return at::native::_sparse_coo_tensor_unsafe(
               indices, self.values(), self.sizes())
        ._coalesced_(true);
  }
  AT_ERROR(
      "sparse_compressed_to_sparse expected SparseCsr or SparseCsc layout but got ",
      self.layout());
}

Tensor sparse_compressed_to_sparse(const Tensor& self) {
  return sparse_compressed_to_sparse(self, 2);
}

// Sparse layout conversions End

Tensor to_meta(const Tensor& tensor) {
  auto out = at::native::empty_strided_meta_symint(tensor.sym_sizes(), tensor.sym_strides(), \
/*dtype=*/c10::make_optional(tensor.scalar_type()), /*layout=*/c10::make_optional(tensor.layout()), \
/*device=*/c10::make_optional(c10::Device(c10::kMeta)), /*pin_memory=*/c10::nullopt);
  // needs to handle wrapped numbers, so dtype promotion works properly.
  if (tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
    out.unsafeGetTensorImpl()->set_wrapped_number(true);
  }
  return out;
}
c10::optional<Tensor> to_meta(const c10::optional<Tensor>& tensor) {
  if (tensor.has_value()) {
    return to_meta(*tensor);
  }
  return c10::nullopt;
}

std::vector<Tensor> to_meta(at::ITensorListRef t_list) {
  std::vector<Tensor> outs;
  outs.reserve(t_list.size());
  for (const auto& tensor : t_list) {
    outs.push_back(to_meta(tensor));
  }
  return outs;
}
} // namespace native
} // namespace at
