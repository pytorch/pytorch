// Basic functions on sparse tensors

#include <ATen/ATen.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/Layout.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/SparseCsrTensorImpl.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/SparseTensorImpl.h>

namespace at {
namespace native {

using namespace at::sparse_csr;

// Construction of CSR tensors.
SparseCsrTensor new_csr_tensor(const TensorOptions& options) {
  // TODO: remove this comment after enabling autograd support for CSR tensor
  // constructor.
  // TORCH_INTERNAL_ASSERT(impl::variable_excluded_from_dispatch());
  TORCH_INTERNAL_ASSERT(options.layout() == kSparseCsr);
  DispatchKey dispatch_key;

  if (options.device().is_cuda()) {
    dispatch_key = DispatchKey::SparseCsrCUDA;
  } else {
    TORCH_INTERNAL_ASSERT(options.device().is_cpu());
    dispatch_key = DispatchKey::SparseCsrCPU;
  }

  return detail::make_tensor<SparseCsrTensorImpl>(
      DispatchKeySet(dispatch_key), options.dtype());
}

// TODO: This constructor should probably use an ATen abstract method in order
// to make autograd dispatch available for the CSR constructor. See the relevant
// note in native_functions.yaml.
Tensor _sparse_csr_tensor(
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
  TORCH_CHECK(
      options.layout() == kSparseCsr,
      "expected sparse CSR layout, but got layout ",
      options.layout());

  AT_DISPATCH_INDEX_TYPES(crow_indices.scalar_type(), "csr_construct_check", [&] {
    auto crow_indices_accessor = crow_indices.accessor<index_t, 1>();
    TORCH_CHECK(
        crow_indices_accessor[crow_indices.numel() - 1] <= col_indices.numel(),
        "last value of crow_indices should be less than length of col_indices.");
    TORCH_CHECK(
        crow_indices_accessor[0] == 0, "0th value of crow_indices must be 0.");
  });

  TORCH_CHECK(
      crow_indices.dim() == 1,
      "crow_indices must have dim=1 but got crow_indices.dim()=",
      crow_indices.dim());
  TORCH_CHECK(
      col_indices.dim() == 1,
      "col_indices must have dim=1 but got col_indices.dim()=",
      col_indices.dim());
  TORCH_CHECK(
      values.dim() == 1,
      "values must have dim=1 but got values.dim()=",
      values.dim());

  TORCH_CHECK(
      (crow_indices.numel() - 1) == size[0],
      "crow_indices.numel() must be size(0) + 1, but got: ",
      crow_indices.numel());

  SparseCsrTensor self = new_csr_tensor(options);
  get_sparse_csr_impl(self)->resize_and_clear_(values.numel(), size);
  get_sparse_csr_impl(self)->set_member_tensors(
      crow_indices, col_indices, values);
  return self;
}

Tensor _sparse_csr_tensor(
    const Tensor& crow_indices,
    const Tensor& col_indices,
    const Tensor& values,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory) {
  // See [Note: hacky wrapper removal for TensorOptions]
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  TORCH_CHECK(
      options.layout() == kSparseCsr,
      "expected sparse CSR layout, but got layout ",
      options.layout());
  TORCH_CHECK(crow_indices.numel() >= 1, "expected crow_indices.numel() >= 1, but got ",
              crow_indices.numel());
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  std::array<int64_t, 2> size;

  if (col_indices.numel() > 0) {
    size[0] = crow_indices.numel() - 1;
    Tensor max_col_indices = std::get<0>(col_indices.max(0, false));

    AT_DISPATCH_INDEX_TYPES(crow_indices.scalar_type(), "csr_construct_check", [&] {
      auto crow_indices_accessor = crow_indices.accessor<index_t, 1>();
      TORCH_CHECK(
          crow_indices_accessor[crow_indices.numel() - 1] <= col_indices.numel(),
          "last value of crow_indices should be less than length of col_indices.");
      TORCH_CHECK(
          crow_indices_accessor[0] == 0, "0th value of crow_indices must be 0.");

      size[1] = *max_col_indices.data_ptr<index_t>() + 1;
    });
  } else {
    size[0] = 0;
    size[1] = 0;
  }

  return at::_sparse_csr_tensor(
      crow_indices, col_indices, values, size, options);
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
      "resize_as_sparse_csr_: layout for self and src must be sparse_csr but got self, src: ",
      self.layout(),
      src.layout());
  if (!_is_same_size_as_sparse_csr(self, src)) {
    get_sparse_csr_impl(self)->resize_as_sparse_csr_tensor_(src);
  }
  return self;
}

} // namespace native
} // namespace at
