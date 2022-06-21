#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <c10/util/Optional.h>
#include <ATen/quantized/Quantizer.h>
#include <ATen/Parallel.h>

#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <ATen/SparseTensorUtils.h>
#include <ATen/native/IndexingUtils.h>

namespace at {
namespace native {

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
  if (self.is_sparse_csr()) {
    TORCH_CHECK(
        memory_format == MemoryFormat::Preserve,
        "sparse_csr only supports memory format Preserve, but got ",
        memory_format,
        " instead.");

    auto new_values = at::native::to(
        self.values(),
        dtype,
        c10::kStrided, // values are strided
        device,
        pin_memory,
        non_blocking,
        true, // force copy since we're in _to_copy
        memory_format);

    auto new_crow_indices = at::native::to(
        self.crow_indices(),
        self.crow_indices().scalar_type(), // indices are integral
        c10::kStrided, // indices are strided
        device,
        pin_memory,
        non_blocking,
        true, // force copy since we're in _to_copy
        memory_format);

    auto new_col_indices = at::native::to(
        self.col_indices(),
        self.col_indices().scalar_type(), // indices are integral
        c10::kStrided, // indices are strided
        device,
        pin_memory,
        non_blocking,
        true, // force copy since we're in _to_copy
        memory_format);

    return at::native::_sparse_csr_tensor_unsafe(
        new_crow_indices,
        new_col_indices,
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
      tensor.layout() == c10::kSparseBsr) {
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
  if (self.layout() == kSparseBsr) {
    TORCH_CHECK(self.dim() == 2, "Can only convert 2D SparseBsr to Strided.");
    Tensor indices = at::_convert_indices_from_csr_to_coo(
        self.crow_indices(), self.col_indices(), false, false);
    auto values = self.values();
    int64_t blocksize[2] = {values.size(-2), values.size(-1)};
    DimVector expanded_size(
        {self.size(0) / blocksize[0],
         self.size(1) / blocksize[1],
         blocksize[0],
         blocksize[1]});
    // We make use of COO dense dimensions here to use the COO to dense format
    // conversion.
    auto self_coo =
        at::native::_sparse_coo_tensor_unsafe(indices, values, expanded_size)
            .coalesce();
    auto dense = self_coo.to_dense();
    // Here we are untiling the result.
    dense = dense.transpose(1, 2);
    dense = dense.reshape({self.size(0), self.size(1)});
    return dense;
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

// Sparse layout conversions Start

Tensor dense_to_sparse_csr(const Tensor& self) {
  return self.to_sparse().to_sparse_csr();
}

Tensor dense_to_sparse_csc(const Tensor& self) {
  return self.to_sparse().to_sparse_csc();
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
      at::native::arange(not_zero_mask.size(-2), index_dtype, kStrided, index_device)
          .view({not_zero_mask.size(-2), 1})
          .expand_as(not_zero_mask)
          .masked_select(not_zero_mask);
  return std::pair<Tensor, Tensor>(col_indices, row_indices);
}

Tensor dense_to_sparse_bsr(const Tensor& self, IntArrayRef blocksize) {
  TORCH_CHECK(self.dim() == 2, "Can only covert 2D Tensor to BSR.");
  TORCH_CHECK(
      blocksize[0] > 0 && blocksize[1] > 0,
      "blocksize needs to be non zero, but got ",
      blocksize);
  TORCH_CHECK(
      self.size(0) % blocksize[0] == 0,
      "Tensor size(0) ",
      self.size(0),
      " needs to be divisible by blocksize[0] ",
      blocksize[0]);
  TORCH_CHECK(
      self.size(1) % blocksize[1] == 0,
      "Tensor size(1) ",
      self.size(1),
      " needs to be divisible by blocksize[1] ",
      blocksize[1]);
  auto block_size_0 = self.size(0) / blocksize[0];

  auto values = _tile_tensor(self, blocksize);
  auto not_zero_mask = _tile_tensor((self != 0), blocksize);
  // Find tiles that have at least 1 non-zero value in them.
  not_zero_mask = not_zero_mask.any(-1).any(-1);
  Tensor col_indices;
  Tensor row_indices;
  std::tie(col_indices, row_indices) =
      _not_zero_mask_to_col_row_indices(not_zero_mask, at::kLong, not_zero_mask.device());
  Tensor crow_indices = at::_convert_indices_from_coo_to_csr(
      row_indices.view({-1}), block_size_0, false /* out_int32 */);
  values = values.reshape({-1, values.size(-2), values.size(-1)});
  not_zero_mask = not_zero_mask.reshape({-1});
  // TODO: masked_select does not support some form of broadcasting, so we're
  // using the mask to construct indices that are then passed into index_select.
  // This isn't ideal.
  values = values.index_select(
      0,
      at::native::arange(not_zero_mask.numel(), at::kLong, kStrided, not_zero_mask.device())
          .masked_select(not_zero_mask));

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
  AT_ERROR(
      "Conversion from ", self.layout(), " to SparseBsc is currently not supported.");
  return self;
}

Tensor sparse_compressed_to_sparse_csr(const Tensor& self) {
  if (self.layout() == kSparseCsc) {
    TORCH_CHECK(
        self.dim() == 2,
        "Expected self to be of dimension 2, but got ",
        self.dim(),
        ".");
    auto sizes = self.sizes();
    auto ccol_indices = self.ccol_indices();
    auto row_indices = self.row_indices();
    auto values = self.values();

    // convert CSC indices to COO indices and swap its rows
    const bool out_int32 = ccol_indices.scalar_type() == ScalarType::Int;
    Tensor indices_transposed = _convert_indices_from_csr_to_coo(
        ccol_indices, row_indices, out_int32, true);

    // sort transposed indices
    auto indices_scalar =
        at::sparse::flatten_indices(indices_transposed, {sizes[0], sizes[1]});
    auto indicesPermutation = std::get<1>(indices_scalar.sort(0));
    auto indices_transposed_sorted =
        indices_transposed.index_select(1, indicesPermutation);

    // construct a CSR tensor
    auto new_row_indices = indices_transposed_sorted.select(0, 0);
    auto new_col_indices = indices_transposed_sorted.select(0, 1);
    auto new_values = values.index_select(0, indicesPermutation);
    Tensor new_crow_indices =
        _convert_indices_from_coo_to_csr(new_row_indices, sizes[0], out_int32);

    return _sparse_csr_tensor_unsafe(
        new_crow_indices,
        new_col_indices,
        new_values,
        {sizes[0], sizes[1]},
        new_values.scalar_type(),
        c10::kSparseCsr,
        new_values.device());
  }
  if (self.layout() == kSparseCsr) {
    // Just returning self doesn't work
    // RuntimeError: t.use_count() <= 1 INTERNAL ASSERT FAILED at
    // "../torch/csrc/autograd/autograd_not_implemented_fallback.cpp":152,
    // please report a bug to PyTorch. aten::to_sparse_csr
    return at::native::_sparse_csr_tensor_unsafe(
        self.crow_indices(),
        self.col_indices(),
        self.values(),
        self.sizes(),
        self.scalar_type(),
        c10::kSparseCsr,
        self.device());
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
  TORCH_CHECK(
      self.is_sparse_csr(),
      "Can only convert CSR to SparseBsr, but got ",
      self.layout(),
      " instead.");
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
          self_values.device()),
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

Tensor sparse_compressed_to_sparse_bsc(const Tensor& self, IntArrayRef blocksize) {
  AT_ERROR(
      "Conversion from ", self.layout(), " to SparseBsc is currently not supported.");
  return self;
}

Tensor sparse_compressed_to_sparse_csc(const Tensor& self) {
  if (self.layout() == kSparseCsc) {
    // Based on to_sparse_csr just returning self doesn't work
    return _sparse_csc_tensor_unsafe(
        self.ccol_indices(),
        self.row_indices(),
        self.values(),
        self.sizes(),
        self.scalar_type(),
        c10::kSparseCsc,
        self.device());
  }
  AT_ERROR(
      "Conversion from ", self.layout(), " to SparseCsc is currently not supported.");
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
  auto out = at::native::empty_strided_meta(tensor.sizes(), tensor.strides(), \
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

std::vector<Tensor> to_meta(const at::TensorList& t_list) {
  std::vector<Tensor> outs;
  outs.reserve(t_list.size());
  for (const auto& i : c10::irange(t_list.size())) {
    outs.push_back(to_meta(t_list[i]));
  }
  return outs;
}

}} // namespace at::native
