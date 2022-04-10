// Basic functions on sparse tensors
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/Layout.h>
#include <ATen/Parallel.h>
#include <ATen/SparseCsrTensorImpl.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/native/LinearAlgebraUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_nnz_native.h>
#include <ATen/ops/_sparse_csr_tensor_unsafe_native.h>
#include <ATen/ops/_validate_sparse_csr_tensor_args_native.h>
#include <ATen/ops/clone_native.h>
#include <ATen/ops/col_indices_native.h>
#include <ATen/ops/copy_native.h>
#include <ATen/ops/crow_indices_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like_native.h>
#include <ATen/ops/empty_native.h>
#include <ATen/ops/resize_as_sparse_native.h>
#include <ATen/ops/resize_native.h>
#include <ATen/ops/sparse_csr_tensor_native.h>
#include <ATen/ops/values_native.h>
#endif

namespace at {
namespace native {

using namespace at::sparse_csr;

namespace {

Layout get_layout_from_members(const Tensor& crow_indices, const Tensor& col_indices, const Tensor& values) {
  if (values.dim() > 1) {
    return kSparseBsr;
  } else {
    return kSparseCsr;
  }
}

} // end anonymous namespace

void _validate_sparse_csr_tensor_args(const Tensor& crow_indices, const Tensor& col_indices, const Tensor& values, IntArrayRef size, c10::optional<Layout> layout) {
  Layout layout_ = layout.value_or(get_layout_from_members(crow_indices, col_indices, values));

  const char* LAYOUT = (layout_ == kSparseCsr ? "CSR" : (layout_ == kSparseCsc ? "CSC" : (layout_ == kSparseBsr ? "BSR" : "BSC")));
  const char* CROW_INDICES = ((layout_ == kSparseCsr || layout_ == kSparseBsr) ? "crow_indices" : "ccol_indices");
  const char* COL_INDICES = ((layout_ == kSparseCsr || layout_ == kSparseBsr) ? "col_indices" : "row_indices");

  // Layout Invariants
  TORCH_CHECK(
      col_indices.layout() == kStrided && col_indices.is_contiguous(),
      "expected ", COL_INDICES, " to be a strided and contiguous tensor");

  TORCH_CHECK(
      crow_indices.layout() == kStrided && crow_indices.is_contiguous(),
      "expected ", CROW_INDICES ," to be a strided and contiguous tensor");

  TORCH_CHECK(
      values.layout() == kStrided && values.is_contiguous(),
      "expected values to be a strided and contiguous tensor");

  // Shape and Strides invariants
  switch (layout_) {
  case kSparseCsr:
  case kSparseCsc:
    TORCH_CHECK(
                size.size() >= 2,
                "size of a batched ", LAYOUT, " tensor must have length >= 2, but got: ",
                size.size());
    TORCH_CHECK(
                crow_indices.dim() >= 1,
                CROW_INDICES, " must have dim >= 1 but got crow_indices.dim() = ",
                crow_indices.dim());
    TORCH_CHECK(
                col_indices.dim() >= 1,
                COL_INDICES, " must have dim >= 1 but got col_indices.dim() = ",
                col_indices.dim());
    TORCH_CHECK(
                values.dim() >= 1,
                "values must have dim >= 1 but got values.dim() = ",
                values.dim());
    break;
  case kSparseBsr:
  case kSparseBsc:
    TORCH_CHECK(false, "_validate_sparse_csr_tensor_args: layout ", layout_, " is not yet supported");  // TODO
    break;
  default:
    TORCH_CHECK(false, "_validate_sparse_csr_tensor_args: layout ", layout_, " is not supported");
  }

  TORCH_CHECK(
      crow_indices.dim() == col_indices.dim(),
      "Number of dimensions of crow_indices and col_indices must be the same.");
  TORCH_CHECK(
      crow_indices.dim() == values.dim(),
      "Number of dimensions of indices and values must be the same.");
  TORCH_CHECK(
      static_cast<size_t>(crow_indices.dim()) == size.size() - 1,
      "Number of dimensions of indices must be one less than the number of dimensions of the provided size.");

  // All batch sizes must be the same
  auto batch_size = size.slice(0, size.size() - 2);
  auto crow_indices_batch_size = crow_indices.sizes().slice(0, crow_indices.dim() - 1);
  auto col_indices_batch_size = col_indices.sizes().slice(0, col_indices.dim() - 1);
  auto values_batch_size = values.sizes().slice(0, values.dim() - 1);
  TORCH_CHECK(
      batch_size == crow_indices_batch_size &&
      batch_size == col_indices_batch_size &&
      batch_size == values_batch_size,
      "All batch dimensions of the provided size (", batch_size, "), indices (",
      crow_indices_batch_size,", ", col_indices_batch_size, "), and values (",
      values_batch_size,") must be the same.");
  // Note, this check also enforces `crow_indices.size(-1) >= 1`
  TORCH_CHECK(
              crow_indices.size(-1) == (size[size.size() - 2] + 1),  // TODO: BSR/BSC
              CROW_INDICES, ".size(-1) must be equal to size[-2] + 1 (that is ", size[size.size() - 2] + 1, "), but got: ",
              crow_indices.size(-1));
  TORCH_CHECK(
      col_indices.numel() == values.numel(),
      COL_INDICES, " and values must have the same number of elements, but got ", COL_INDICES, ".numel(): ",
      col_indices.numel(),
      ", values.numel(): ",
      values.numel());

  // Indices invariants
  AT_DISPATCH_INDEX_TYPES(crow_indices.scalar_type(), "csr_construct_check", [&] {
    Tensor crow_indices_cpu = crow_indices.to(kCPU);
    auto crow_indices_data_ptr = crow_indices_cpu.data_ptr<index_t>();
    auto batch_stride = crow_indices_cpu.dim() >= 2 ? crow_indices_cpu.stride(-2) : 0;
    switch (layout_) {
    case kSparseCsr:
    case kSparseCsc:
      for (const auto batch_id : c10::irange(batchCount(crow_indices_cpu))) {
        TORCH_CHECK(
                    crow_indices_data_ptr[batch_id*batch_stride] == 0,
                    "(Batch element ", batch_id, ") ",
                    ": 0th value of ", CROW_INDICES, " must be 0, but it is ", crow_indices_data_ptr[batch_id*batch_stride]);
        TORCH_CHECK(
                    crow_indices_data_ptr[batch_id*batch_stride + crow_indices.size(-1) - 1] == col_indices.size(-1),
                    "(Batch element ", batch_id, ") ",
                    "last value of ", CROW_INDICES, " should be equal to the length of col_indices.");
        for (int i =  1; i <= size[size.size() - 2]; i++) {
          TORCH_CHECK(
                      crow_indices_data_ptr[batch_id*batch_stride + i - 1] <= crow_indices_data_ptr[batch_id*batch_stride + i],
                      "(Batch element ", batch_id, ") ",
                      "at position i = ", i, ", the condition ", CROW_INDICES, "[i - 1] <= ", CROW_INDICES, "[i] fails");
        }
      }
      if (col_indices.numel() > 0) {
        TORCH_CHECK(0 <= col_indices.min().item<index_t>(), COL_INDICES, ".min() should be greater or equal to zero");
        TORCH_CHECK(size[size.size() - 1] > col_indices.max().item<index_t>(), "size[-1] should be greater than ", COL_INDICES, ".max()");
      }
      break;
    case kSparseBsr:
    case kSparseBsc:
      TORCH_CHECK(false, "_validate_sparse_csr_tensor_args: layout ", layout_, " is not yet supported");  // TODO
    default:
      TORCH_CHECK(false, "_validate_sparse_csr_tensor_args: layout ", layout_, " is not supported");
    }
  });

  // CSR Type Invariants
  auto crow_indices_type = crow_indices.scalar_type();
  auto col_indices_type = col_indices.scalar_type();
  TORCH_CHECK(
      crow_indices_type == col_indices_type,
      "both ", CROW_INDICES, " and ", COL_INDICES, " should have the same type.");
  TORCH_CHECK(
      crow_indices_type == kInt || crow_indices_type == kLong,
      CROW_INDICES, " and ", COL_INDICES, " must be an int32 or int64 type, but got: ",
      crow_indices_type);

  // CSR Device Invariants
  TORCH_CHECK(
      col_indices.get_device() == crow_indices.get_device(),
      CROW_INDICES, " and ", COL_INDICES, " devices (",
      crow_indices.get_device(),
      ", ",
      col_indices.get_device(),
      ") must match");
  TORCH_CHECK(
      crow_indices.get_device() == values.get_device(),
      "device of ", CROW_INDICES, " (",
      crow_indices.get_device(),
      ") must match device of values (",
      values.get_device(),
      ")");
  TORCH_CHECK(
      values.device().type() == kCPU || values.device().type() == kCUDA,
      "device type of values (",
      values.device().type(),
      ") must be CPU or CUDA");
}

// Construction of CSR tensors.
SparseCsrTensor new_csr_tensor(const TensorOptions& options) {
  // TODO: remove this comment after enabling autograd support for CSR tensor
  // constructor.
  // TORCH_INTERNAL_ASSERT(impl::variable_excluded_from_dispatch());
  DispatchKey dispatch_key;

  TORCH_CHECK_NOT_IMPLEMENTED(
    options.device().type() == kCPU || options.device().type() == kCUDA,
     "Could not run 'sparse_csr_tensor' from the '", options.device(), "' device.)");

  if (options.device().is_cuda()) {
    dispatch_key = DispatchKey::SparseCsrCUDA;
  } else {
    dispatch_key = DispatchKey::SparseCsrCPU;
  }

  return detail::make_tensor<SparseCsrTensorImpl>(
    DispatchKeySet(dispatch_key), options.layout(), options.dtype());
}

Tensor _sparse_csr_tensor_unsafe(const Tensor& crow_indices, const Tensor& col_indices,
    const Tensor& values,
    IntArrayRef size,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,  // unsafe requires explicit layout
    c10::optional<Device> device,
    c10::optional<bool> pin_memory) {
  TORCH_CHECK(layout.has_value(), "_sparse_csr_tensor_unsafe requires explicit layout specification");
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  SparseCsrTensor self = new_csr_tensor(options);
  get_sparse_csr_impl(self)->set_member_tensors(crow_indices, col_indices, values, size);
  return self;
}

// TODO: This constructor should probably use an ATen abstract method in order
// to make autograd dispatch available for the CSR constructor. See the relevant
// note in native_functions.yaml.
Tensor sparse_csr_tensor(
    const Tensor& crow_indices,
    const Tensor& col_indices,
    const Tensor& values,
    IntArrayRef size,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory) {
  Layout layout_ = layout.value_or(get_layout_from_members(crow_indices, col_indices, values));

  // See [Note: hacky wrapper removal for TensorOptions]
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout_).device(device).pinned_memory(pin_memory);

  at::native::_validate_sparse_csr_tensor_args(crow_indices, col_indices, values, size, layout_);

  return at::native::_sparse_csr_tensor_unsafe(
      crow_indices,
      col_indices,
      values,
      size,
      optTypeMetaToScalarType(options.dtype_opt()),
      layout_,
      options.device_opt(),
      options.pinned_memory_opt());
}

Tensor sparse_csr_tensor(
    const Tensor& crow_indices,
    const Tensor& col_indices,
    const Tensor& values,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory) {
  Layout layout_ = layout.value_or(get_layout_from_members(crow_indices, col_indices, values));
  // See [Note: hacky wrapper removal for TensorOptions]
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);
  // TODO: extend size for BSR|BSC
  auto size = DimVector(IntArrayRef(col_indices.sizes().data(), col_indices.dim() - 1));
  switch (layout_) {
  case kSparseCsr:
    if (col_indices.size(-1) > 0) {
      size.push_back(crow_indices.size(-1) - 1);

    } else {
      size.push_back(0);
    }
    AT_DISPATCH_INDEX_TYPES(col_indices.scalar_type(), "csr_construct_check", [&] {
                                                                                size.push_back(col_indices.max().item<index_t>() + 1);
                                                                              });
    break;
  case kSparseCsc:
    AT_DISPATCH_INDEX_TYPES(col_indices.scalar_type(), "csr_construct_check", [&] {
                                                                                size.push_back(col_indices.max().item<index_t>() + 1);
                                                                              });
    if (col_indices.size(-1) > 0) {
      size.push_back(crow_indices.size(-1) - 1);

    } else {
      size.push_back(0);
    }
    break;
  case kSparseBsr:
  case kSparseBsc:
    TORCH_CHECK(false, "sparse_csr_tensor: layout ", layout_, " is not yet supported");  // TODO
    break;
  default:
    TORCH_CHECK(false, "sparse_csr_tensor: layout ", layout_, " is not supported");
  }

  at::native::_validate_sparse_csr_tensor_args(crow_indices, col_indices, values, size, layout_);

  return at::native::_sparse_csr_tensor_unsafe(
      crow_indices,
      col_indices,
      values,
      size,
      optTypeMetaToScalarType(options.dtype_opt()),
      layout_,
      options.device_opt(),
      options.pinned_memory_opt());
}

Tensor empty_sparse_csr(
    IntArrayRef size,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<MemoryFormat> optional_memory_format) {
  check_size_nonnegative(size);
  Layout layout_ = layout.value_or((size.size() == 2 ? kSparseCsr : kSparseBsr));  // TODO: this is wrong..
  const char* LAYOUT = (layout_ == kSparseCsr ? "CSR" : (layout_ == kSparseCsc ? "CSC" : (layout_ == kSparseBsr ? "BSR" : "BSC")));

  TensorOptions options = TensorOptions().dtype(ScalarType::Long).layout(Layout::Strided).device(device).pinned_memory(pin_memory);
  Tensor crow_indices;
  Tensor col_indices;
  Tensor values;

  switch (layout_) {
  case kSparseCsr:
  case kSparseCsc:
    {
      TORCH_CHECK(size.size() >= 2, "torch.empty: Only batched sparse ", LAYOUT, " matrices are supported, but got size ", size);
      auto rows = size[size.size() - ((layout == kSparseCsr) ?  2 : 1)];
      auto crow_indices_size = DimVector(size.slice(0, size.size() - 2));
      crow_indices_size.push_back(rows + 1);
      auto col_indices_values_size = DimVector(size.slice(0, size.size() - 2));
      int64_t nnz = 0;
      col_indices_values_size.push_back(nnz);
      crow_indices = at::empty(crow_indices_size, options);
      col_indices = at::empty(col_indices_values_size, options);
      values = at::empty(col_indices_values_size, options.dtype(dtype));
    }
    break;
  case kSparseBsr:
  case kSparseBsc:
    TORCH_CHECK(size.size() >= 2, "torch.empty: At least 2D sparse ", LAYOUT, " tensors are supported.");
    TORCH_CHECK(false, "empty_sparse_csr: layout ", layout_, " is not yet supported");  // TODO: replace with implementation
    break;
  default:
    TORCH_CHECK(false, "empty_sparse_csr: layout ", layout_, " is not supported");
  }

  return at::native::_sparse_csr_tensor_unsafe(
      crow_indices,
      col_indices,
      values,
      size,
      dtype,
      layout_,
      device,
      pin_memory);
}

const Tensor& resize_sparse_csr_(
    const Tensor& self,
    IntArrayRef size,
    c10::optional<MemoryFormat> optional_memory_format) {
  check_size_nonnegative(size);
  Layout layout_ = self.layout();
  const char* LAYOUT = (layout_ == kSparseCsr ? "CSR" : (layout_ == kSparseCsc ? "CSC" : (layout_ == kSparseBsr ? "BSR" : "BSC")));
  switch (layout_) {
  case kSparseCsr:
  case kSparseCsc:
    TORCH_CHECK(size.size() >= 2, "torch.resize_: Only batched sparse ", LAYOUT, " tensors are supported, but got size ", size);
    break;
  case kSparseBsr:
  case kSparseBsc:
    TORCH_CHECK(size.size() >= 2, "torch.resize_: At least 2D sparse ", LAYOUT, " tensors are supported, but got size ", size);  // FIXME
    break;
  default:
    TORCH_CHECK(false, "resize_sparse_csr_: layout ", layout_, " is not supported");
  }

  auto col_index = size.size() - ((layout_ == kSparseCsr || layout_ == kSparseBsr) ? 1 : 2);
  const char* COLUMNS = ((layout_ == kSparseCsr || layout_ == kSparseBsr) ? "columns" : "rows");
  TORCH_CHECK(
      self.size(col_index) <= size[col_index],
      "torch.resize_: Resizing ", COLUMNS, " of sparse ", LAYOUT, " tensors to a smaller value is not supported. ",
      "The original number of ", COLUMNS, " is ",
      self.size(col_index),
      " while the requested new number of ", COLUMNS, " is ", size[col_index], ".");
  get_sparse_csr_impl(self)->resize_(self._nnz(), size);
  return self;
}

Tensor& copy_sparse_csr_(Tensor& self, const Tensor& src, bool non_blocking) {
  TORCH_CHECK(
      self.sizes() == src.sizes(),
      "copy_sparse_csr_: only same size tensors are supported.");
  switch (src.layout()) {
  case kSparseCsr:
  case kSparseCsc:
  case kSparseBsr:
  case kSparseBsc:
    TORCH_CHECK(
                self.layout() == src.layout(),  // TODO: support CSR->BSR, CSC->BSC
                "copy between different layouts is not supported. Found self type = ",
                self.toString(),
                " and src type = ",
                src.toString());
    break;
  default:
    TORCH_CHECK(false, "copy_sparse_csr_: layout ", src.layout(), " is not supported");
  }

  TORCH_CHECK(
      self._nnz() == src._nnz(),
      "copy_sparse_csr_: only tensors with the same number of specified elements are supported.");
  self.crow_indices().copy_(src.crow_indices(), non_blocking);
  self.col_indices().copy_(src.col_indices(), non_blocking);
  self.values().copy_(src.values(), non_blocking);
  get_sparse_csr_impl(self)->set_layout(src.layout());
  return self;
}

// Access members of CSR tensors.
int64_t _nnz_sparse_csr(const SparseCsrTensor& self) {
  return get_sparse_csr_impl(self)->nnz();
}

Tensor values_sparse_csr(const Tensor& self) {
  return get_sparse_csr_impl(self)->values().alias();
}

Tensor crow_indices_sparse_csr(const Tensor& self) {
  return get_sparse_csr_impl(self)->crow_indices().alias();
}

Tensor col_indices_sparse_csr(const Tensor& self) {
  return get_sparse_csr_impl(self)->col_indices().alias();
}

bool _is_same_size_as_sparse_csr(
    const SparseCsrTensor& self,
    const SparseCsrTensor& src) {
  return self.sizes().equals(src.sizes());
}

const SparseCsrTensor& resize_as_sparse_csr_(
    const SparseCsrTensor& self,
    const SparseCsrTensor& src) {
  switch(self.layout()) {
  case kSparseCsr:
  case kSparseCsc:
  case kSparseBsr:
  case kSparseBsc:
    TORCH_CHECK(
                src.layout() == self.layout(),
                "resize_as_sparse_csr_: layout for self and src must be match but got ",
                self.layout(),
                " for self, and ",
                src.layout(),
                " for src");
    break;
  default:
    TORCH_CHECK(false, "resize_as_sparse_csr_: layout ", self.layout(), " is not supported");
  }
  if (!_is_same_size_as_sparse_csr(self, src)) {
    get_sparse_csr_impl(self)->resize_as_sparse_csr_tensor_(src);
  }
  return self;
}

SparseCsrTensor clone_sparse_csr(
    const SparseCsrTensor& self,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  TORCH_CHECK(
      !optional_memory_format.has_value(),
      "unsupported memory format option ",
      optional_memory_format.value());
  TensorOptions options = self.options();
  return at::native::_sparse_csr_tensor_unsafe(
                                               self.crow_indices().clone(),
                                               self.col_indices().clone(),
                                               self.values().clone(),
                                               self.sizes(),
                                               optTypeMetaToScalarType(options.dtype_opt()),
                                               options.layout(),
                                               options.device_opt(),
                                               options.pinned_memory_opt());
}

Tensor empty_like_sparse_csr(
    const Tensor& self,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  TensorOptions options_ = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);
  TensorOptions options =
      self.options()
          .merge_in(options_)
          .merge_memory_format(optional_memory_format);

  switch(options.layout()) {
  case kSparseCsr:
  case kSparseCsc:
  case kSparseBsr:
  case kSparseBsc:
    return at::native::_sparse_csr_tensor_unsafe(
                                                 self.crow_indices().clone(),
                                                 self.col_indices().clone(),
                                                 at::empty(self.values().sizes(), options.layout(kStrided)),
                                                 self.sizes(),
                                                 dtype,
                                                 options.layout(),
                                                 device);
    break;
  // TODO: kSparse
  case kStrided:
    return at::native::empty_like(self, dtype, layout, device, pin_memory, optional_memory_format);
    break;
  default:
    TORCH_CHECK(false, "empty_like_sparse_csr: layout ", options.layout(), " is not supported");
  }
}

} // namespace native
} // namespace at
