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
#include <ATen/ops/_convert_indices_from_csr_to_coo.h>
#include <ATen/ops/_nnz_native.h>
#include <ATen/ops/_sparse_csr_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_coo_tensor_unsafe_native.h>
#include <ATen/ops/_validate_sparse_compressed_tensor_args_native.h>
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
#include <ATen/ops/select_native.h>
#include <ATen/ops/sparse_csr_tensor_native.h>
#include <ATen/ops/values_native.h>
#endif

namespace at {
namespace native {

using namespace at::sparse_csr;

namespace {


} // end anonymous namespace

void _validate_sparse_compressed_tensor_args_worker(const Tensor& compressed_indices, const Tensor& plain_indices, const Tensor& values, const IntArrayRef size, const Layout& layout) {

  // Layout must be Sparse Compressed
  AT_DISPATCH_ALL_SPARSE_COMPRESSED_LAYOUTS(layout, "validate_sparse_compressed_tensor_args", [&]{});

  const std::string layout_name = layoutToString(layout, /*upper=*/ true);
  const std::string compressed_indices_name = compressedIndicesName(layout);
  const std::string plain_indices_name = plainIndicesName(layout);

  // Layout Invariants
  TORCH_CHECK(
      plain_indices.layout() == kStrided && plain_indices.is_contiguous(),
      "expected ", plain_indices_name, " to be a strided and contiguous tensor");

  TORCH_CHECK(
      compressed_indices.layout() == kStrided && compressed_indices.is_contiguous(),
      "expected ", compressed_indices_name ," to be a strided and contiguous tensor");

  TORCH_CHECK(
      values.layout() == kStrided && values.is_contiguous(),
      "expected values to be a strided and contiguous tensor");

  // Shape and Strides invariants
  TORCH_CHECK(
              size.size() >= 2,
              "size of a batched ", layout_name, " tensor must have length >= 2, but got: ",
              size.size());
  TORCH_CHECK(
              compressed_indices.dim() >= 1,
              compressed_indices_name, " must have dim >= 1 but got ", compressed_indices_name, ".dim() = ",
              compressed_indices.dim());
  TORCH_CHECK(
              plain_indices.dim() >= 1,
              plain_indices_name, " must have dim >= 1 but got ", plain_indices_name, ".dim() = ",
              plain_indices.dim());
  TORCH_CHECK(
              values.dim() >= 1,
              "values must have dim >= 1 but got values.dim() = ",
              values.dim());

  TORCH_CHECK(
      compressed_indices.dim() == plain_indices.dim(),
      "number of dimensions of ", compressed_indices_name, " and ", plain_indices_name, " must be the same.");

  AT_DISPATCH_PLAIN_SPARSE_COMPRESSED_LAYOUTS(
      layout, "validate_sparse_compressed_tensor_args",
      [&] {
        TORCH_CHECK(
                    compressed_indices.dim() == values.dim(),
                    "number of dimensions of indices and values must be the same.");
      },
      [&] {
        TORCH_CHECK(
                    compressed_indices.dim() + 2 == values.dim(),
                    "number of dimensions of indices must be two less than the number of dimensions of the values.");
      });

  TORCH_CHECK(
      static_cast<size_t>(compressed_indices.dim()) == size.size() - 1,
      "number of dimensions of indices must be one less than the number of dimensions of the provided size.");

  int block_ndim = AT_DISPATCH_PLAIN_SPARSE_COMPRESSED_LAYOUTS(layout, "validate_sparse_compressed_tensor_args", [&]{ return 0; }, [&]{ return 2; });
  IntArrayRef block_size = values.sizes().slice(values.dim() - block_ndim, block_ndim);
  int64_t numel_per_block = AT_DISPATCH_PLAIN_SPARSE_COMPRESSED_LAYOUTS(layout, "validate_sparse_compressed_tensor_args",
                                [&]() -> int64_t { return 1; }, [&]() -> int64_t { return block_size[0] * block_size[1]; });
  int compressed_dim = compressedDimension(layout, size);
  int plain_dim = plainDimension(layout, size);

  // All batch sizes must be the same
  auto batch_size = size.slice(0, size.size() - 2);
  auto compressed_indices_batch_size = compressed_indices.sizes().slice(0, compressed_indices.dim() - 1);
  auto plain_indices_batch_size = plain_indices.sizes().slice(0, plain_indices.dim() - 1);
  auto values_batch_size = values.sizes().slice(0, values.dim() - 1 - block_ndim);
  TORCH_CHECK(
      batch_size == compressed_indices_batch_size &&
      batch_size == plain_indices_batch_size &&
      batch_size == values_batch_size,
      "all batch dimensions of the provided size (", batch_size, "), indices (",
      compressed_indices_batch_size,", ", plain_indices_batch_size, "), and values (",
      values_batch_size,") must be the same.");

  // Note, this check also enforces `compressed_indices.size(-1) >= 1`
  TORCH_CHECK(
              compressed_indices.size(-1) == (size[compressed_dim] + 1),
              compressed_indices_name, ".size(-1) must be equal to size[-", (size.size() - compressed_dim), "] + 1 (that is ",
              size[compressed_dim] + 1, "), but got: ", compressed_indices.size(-1));

  AT_DISPATCH_PLAIN_SPARSE_COMPRESSED_LAYOUTS(layout, "validate_sparse_compressed_tensor_args",
      [&] {
        TORCH_CHECK(
                    plain_indices.numel() == values.numel(),
                    plain_indices_name, " and values must have the same number of elements, but got ", plain_indices_name, ".numel(): ",
                    plain_indices.numel(), ", values.numel(): ", values.numel());
      },
      [&] {
        TORCH_CHECK(
                    plain_indices.numel() * numel_per_block == values.numel(),
                    "number of ", plain_indices_name, " elements must be the same as the number of blocks in values, but got ",
                    plain_indices_name, ".numel() * numel_per_block: ", plain_indices.numel() * numel_per_block,
                    ", values.numel(): ", values.numel(),", numel_per_block: ", numel_per_block);
      });

  // Indices invariants
  AT_DISPATCH_INDEX_TYPES(compressed_indices.scalar_type(), "validate_sparse_compressed_tensor_args",
      [&] {
        Tensor compressed_indices_cpu = compressed_indices.to(kCPU);
        auto compressed_indices_data_ptr = compressed_indices_cpu.data_ptr<index_t>();
        auto batch_stride = compressed_indices_cpu.dim() >= 2 ? compressed_indices_cpu.stride(-2) : 0;

        for (const auto batch_id : c10::irange(batchCount(compressed_indices_cpu))) {
          TORCH_CHECK(
                      compressed_indices_data_ptr[batch_id*batch_stride] == 0,
                      "(Batch element ", batch_id, ") ",
                      ": 0th value of ", compressed_indices_name, " must be 0, but it is ", compressed_indices_data_ptr[batch_id*batch_stride]);
          TORCH_CHECK(
                      compressed_indices_data_ptr[batch_id*batch_stride + compressed_indices.size(-1) - 1] == plain_indices.size(-1),
                      "(Batch element ", batch_id, ") ",
                      "last value of ", compressed_indices_name, " should be equal to the length of ", plain_indices_name, ".");
          for (int i =  1; i <= size[size.size() - 2]; i++) {
            TORCH_CHECK(
                        compressed_indices_data_ptr[batch_id*batch_stride + i - 1] <= compressed_indices_data_ptr[batch_id*batch_stride + i],
                        "(Batch element ", batch_id, ") ",
                        "at position i = ", i, ", the condition ", compressed_indices_name, "[i - 1] <= ", compressed_indices_name, "[i] fails");
          }
        }
        if (plain_indices.numel() > 0) {
          TORCH_CHECK(0 <= plain_indices.min().item<index_t>(), plain_indices_name, ".min() should be greater or equal to zero");
          TORCH_CHECK(size[plain_dim] > plain_indices.max().item<index_t>(), "size[-", (size.size() - plain_dim),"] should be greater than ", plain_indices_name, ".max()");
        }
      });

  // Type Invariants
  auto compressed_indices_type = compressed_indices.scalar_type();
  auto plain_indices_type = plain_indices.scalar_type();
  TORCH_CHECK(
      compressed_indices_type == plain_indices_type,
      "both ", compressed_indices_name, " and ", plain_indices_name, " should have the same type.");
  TORCH_CHECK(
      compressed_indices_type == kInt || compressed_indices_type == kLong,
      compressed_indices_name, " and ", plain_indices_name, " must be an int32 or int64 type, but got: ",
      compressed_indices_type);

  // Device Invariants
  TORCH_CHECK(
      plain_indices.get_device() == compressed_indices.get_device(),
      compressed_indices_name, " and ", plain_indices_name, " devices (",
      compressed_indices.get_device(),
      ", ",
      plain_indices.get_device(),
      ") must match");
  TORCH_CHECK(
      compressed_indices.get_device() == values.get_device(),
      "device of ", compressed_indices_name, " (",
      compressed_indices.get_device(),
      ") must match device of values (",
      values.get_device(),
      ")");
  TORCH_CHECK(
      values.device().type() == kCPU || values.device().type() == kCUDA,
      "device type of values (",
      values.device().type(),
      ") must be CPU or CUDA");

}

void _validate_sparse_compressed_tensor_args(const Tensor& crow_indices, const Tensor& col_indices, const Tensor& values, IntArrayRef size, Layout layout) {
  _validate_sparse_compressed_tensor_args_worker(crow_indices, col_indices, values, size, layout);
}

void _validate_sparse_csr_tensor_args(const Tensor& crow_indices, const Tensor& col_indices, const Tensor& values, IntArrayRef size) {
  _validate_sparse_compressed_tensor_args_worker(crow_indices, col_indices, values, size, kSparseCsr);
}

// Construction of CSR tensors.
SparseCsrTensor new_csr_tensor(const TensorOptions& options) {
  // TODO: remove this comment after enabling autograd support for CSR tensor
  // constructor.
  // TORCH_INTERNAL_ASSERT(impl::variable_excluded_from_dispatch());
  Layout layout = options.layout();
  TORCH_INTERNAL_ASSERT(layout == kSparseCsr);
  DispatchKey dispatch_key;

  TORCH_CHECK_NOT_IMPLEMENTED(
    options.device().type() == kCPU || options.device().type() == kCUDA,
     "Could not run '", "sparse_csr_tensor", "' from the '", options.device(), "' device.)");

  if (options.device().is_cuda()) {
    dispatch_key = DispatchKey::SparseCsrCUDA;
  } else {
    dispatch_key = DispatchKey::SparseCsrCPU;
  }

  return detail::make_tensor<SparseCsrTensorImpl>(
      DispatchKeySet(dispatch_key), layout, options.dtype());
}

Tensor _sparse_csr_tensor_unsafe(const Tensor& crow_indices, const Tensor& col_indices,
    const Tensor& values,
    IntArrayRef size,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory) {

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
  // See [Note: hacky wrapper removal for TensorOptions]
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  at::native::_validate_sparse_csr_tensor_args(crow_indices, col_indices, values, size);

  return at::native::_sparse_csr_tensor_unsafe(
      crow_indices,
      col_indices,
      values,
      size,
      optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(),
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
  // See [Note: hacky wrapper removal for TensorOptions]
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);
  // std::array<int64_t, 2> size = {0, 0};
  auto size = DimVector(IntArrayRef(col_indices.sizes().data(), col_indices.dim() - 1));
  size.push_back(crow_indices.size(-1) - 1);
  size.push_back(col_indices.max().item<int64_t>() + 1);

  at::native::_validate_sparse_csr_tensor_args(crow_indices, col_indices, values, size);

  return at::native::_sparse_csr_tensor_unsafe(
      crow_indices,
      col_indices,
      values,
      size,
      optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(),
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

  TORCH_CHECK(size.size() >= 2, "torch.empty: Only batched sparse CSR matrices are supported, but got size ", size);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(layout == Layout::SparseCsr);

  auto rows = size[size.size() - 2];
  int64_t nnz = 0;

  auto crow_indices_size = DimVector(size.slice(0, size.size() - 2));
  crow_indices_size.push_back(rows + 1);
  auto col_indices_values_size = DimVector(size.slice(0, size.size() - 2));
  col_indices_values_size.push_back(nnz);

  TensorOptions options = TensorOptions().dtype(ScalarType::Long).layout(Layout::Strided).device(device).pinned_memory(pin_memory);
  auto crow_indices = at::empty(crow_indices_size, options);
  auto col_indices = at::empty(col_indices_values_size, options);
  auto values = at::empty(col_indices_values_size, options.dtype(dtype));

  return at::native::_sparse_csr_tensor_unsafe(
      crow_indices,
      col_indices,
      values,
      size,
      dtype,
      layout,
      device,
      pin_memory);
}

const Tensor& resize_sparse_csr_(
    const Tensor& self,
    IntArrayRef size,
    c10::optional<MemoryFormat> optional_memory_format) {
  check_size_nonnegative(size);
  TORCH_CHECK(size.size() >= 2, "torch.resize_: Only batched sparse CSR matrices are supported, but got size ", size);
  TORCH_CHECK(
      self.size(-1) <= size[size.size() - 1],
      "torch.resize_: Resizing columns of sparse CSR tensors to a smaller value is not supported. ",
      "The original number of columns is ",
      self.size(-1),
      " while the requested new number of columns is ", size[size.size() - 1], ".");
  get_sparse_csr_impl(self)->resize_(self._nnz(), size);
  return self;
}

Tensor& copy_sparse_csr_(Tensor& self, const Tensor& src, bool non_blocking) {
  TORCH_CHECK(
      self.sizes() == src.sizes(),
      "copy_sparse_csr_: only same size tensors are supported.");
  TORCH_CHECK(
      self.is_sparse_csr() && src.is_sparse_csr(),
      "copy_sparse_csr_: copy between different layouts is not supported. Found self type = ",
      self.toString(),
      " and src type = ",
      src.toString());
  TORCH_CHECK(
      self._nnz() == src._nnz(),
      "copy_sparse_csr_: only tensors with the same number of specified elements are supported.");
  self.crow_indices().copy_(src.crow_indices(), non_blocking);
  self.col_indices().copy_(src.col_indices(), non_blocking);
  self.values().copy_(src.values(), non_blocking);
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
  TORCH_CHECK(
      src.is_sparse_csr() && self.is_sparse_csr(),
      "resize_as_sparse_csr_: layout for self and src must be sparse_csr but got ",
      self.layout(),
      " for self, and ",
      src.layout(),
      " for src");
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
                                               options.layout_opt(),
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

  if (options.layout() == kSparseCsr) {
    auto result = at::native::_sparse_csr_tensor_unsafe(
        self.crow_indices().clone(),
        self.col_indices().clone(),
        at::empty(self.values().sizes(), options.layout(kStrided)),
        self.sizes(),
        dtype,
        self.layout(),
        device);
    return result;
  } else if (options.layout() == kStrided) {
    return at::native::empty_like(self, dtype, layout, device, pin_memory, optional_memory_format);
  } else {
    TORCH_CHECK(false, "Layout ", options.layout(), " is not supported");
  }
}

Tensor select_sparse_csr(const Tensor& self, int64_t dim, int64_t index) {
  TORCH_INTERNAL_ASSERT(self.is_sparse_csr());
  TORCH_CHECK_INDEX(self.dim() != 0, "select() cannot be applied to a 0-dim tensor.");
  dim = maybe_wrap_dim(dim, self.dim());
  auto size = self.size(dim);
  if (index < -size || index >= size) {
    TORCH_CHECK_INDEX(false, "select(): index ", index, " out of range for tensor of size ",
                   self.sizes(), " at dimension ", dim);
  }
  if (index < 0) {
    index += size;
  }

  TORCH_INTERNAL_ASSERT(dim >= 0 && dim < self.dim());

  auto new_sizes = DimVector(self.sizes());
  new_sizes.erase(new_sizes.begin() + dim);
  auto options = self.options();

  // Selecting batch dimension
  if (dim < self.dim() - 2) {
    return at::native::_sparse_csr_tensor_unsafe(
        self.crow_indices().select(dim, index),
        self.col_indices().select(dim, index),
        self.values().select(dim, index),
        new_sizes,
        optTypeMetaToScalarType(options.dtype_opt()),
        options.layout_opt(),
        options.device_opt(),
        options.pinned_memory_opt());
  } else {
    TORCH_CHECK(self.dim() == 2, "select(): selecting rows or columns is not implemented for batched sparse CSR tensors.")
    // Converting to COO and calling select is slighly slower than operating on the CSR indices directly
    // for constructing a COO vector, however current version is more readable and easier to understand.
    return self.to_sparse().select(dim, index);
  }
}

} // namespace native
} // namespace at
