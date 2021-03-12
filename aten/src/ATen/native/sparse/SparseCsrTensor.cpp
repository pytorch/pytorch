// Basic functions on sparse tensors

#include <ATen/ATen.h>
#include <ATen/Layout.h>
#include <ATen/Parallel.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/SparseCsrTensorImpl.h>
#include <ATen/NativeFunctions.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/SparseCsrTensorUtils.h>

namespace at { namespace native {

using namespace at::sparse;

// Construction of CSR tensors.
SparseTensor new_csr_tensor(const TensorOptions& options) {
  // TODO: remove this comment after enabling autograd support for CSR tensor constructor.
  // TORCH_INTERNAL_ASSERT(impl::variable_excluded_from_dispatch());
  AT_ASSERT(options.layout() == kSparseCsr);
  DispatchKey dispatch_key;
  if (options.device().is_cuda()) {
    dispatch_key = DispatchKey::SparseCsrCUDA;
  } else {
    dispatch_key = DispatchKey::SparseCsrCPU;
  }
  
  return detail::make_tensor<SparseCsrTensorImpl>(
                                                  DispatchKeySet(dispatch_key), options.dtype());
}

// TODO: This constructor should probably use an ATen abstract method in order to make
// autograd dispatch available for the CSR constructor. See the relevant note in native_functions.yaml. 
Tensor sparse_csr_tensor(const Tensor& crow_indices, const Tensor& col_indices, 
                         const Tensor& values, IntArrayRef size,
                         const TensorOptions& options) {
  TORCH_CHECK(!options.has_layout() || options.layout() == kSparseCsr, 
    "expected sparse CSR layout, but got layout ", options.layout());

  AT_DISPATCH_INDEX_TYPES(crow_indices.scalar_type(), "csr_construct_check", [&] {
      auto crow_indices_accessor = crow_indices.accessor<index_t, 1>();
      TORCH_CHECK(crow_indices_accessor[crow_indices.numel() - 1] <= col_indices.numel(),
              "last value of crow_indices should be less than length of col_indices.");
      TORCH_CHECK(crow_indices_accessor[0] == 0,
                  "0th value of crow_indices must be 0.");
  });

  TORCH_CHECK(crow_indices.dim() == 1, "crow_indices must have dim=1 but got crow_indices.dim()=", 
              crow_indices.dim());
  TORCH_CHECK(col_indices.dim() == 1, "col_indices must have dim=1 but got col_indices.dim()=",
              col_indices.dim()); 
  TORCH_CHECK(values.dim() == 1, "values must have dim=1 but got values.dim()=", values.dim());

  TORCH_CHECK((crow_indices.numel() - 1) == size[0], 
    "crow_indices.numel() must be size(0) + 1, but got: ", crow_indices.numel());

  SparseTensor self = new_csr_tensor(options);
  get_sparse_csr_impl(self)->resize_and_clear_(values.numel(), size);
  get_sparse_csr_impl(self)->set_member_tensors(crow_indices, col_indices, values);
  return self;
}

Tensor sparse_csr_tensor(const Tensor& crow_indices, const Tensor& col_indices,
                         const Tensor& values, const TensorOptions& options) {
  TORCH_CHECK(!options.has_layout() || options.layout() == kSparseCsr, 
    "expected sparse CSR layout, but got layout ", options.layout());
  std::vector<int64_t> size(2);

  if (crow_indices.numel() > 0 && col_indices.numel() > 0) {
    size[0] = crow_indices.numel() - 1;
    Tensor max_col_indices = std::get<0>(col_indices.max(0, false));
    size[1] = *max_col_indices.data_ptr<int64_t>() + 1;
  } else {
    size[0] = 0;
    size[1] = 0;
  }

  return at::sparse_csr_tensor(crow_indices, col_indices, values, size, options);
}

// Access members of CSR tensors.
int64_t _nnz_sparse_csr(const SparseTensor& self) {
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

bool _is_same_size_as_sparse_csr(const SparseTensor& self, const SparseTensor& src) {
  return self.dim() == src.dim() && self.sizes().equals(src.sizes());
}

SparseTensor& resize_as_sparse_csr_(SparseTensor& self, const SparseTensor& src) {
  TORCH_CHECK(src.is_sparse_csr() && self.is_sparse_csr(), "resize_as_sparse_csr_: layout for self and src must be sparse_csr but got self, src: ",
    self.layout(), src.layout());
  if (!_is_same_size_as_sparse_csr(self, src)) {
    get_sparse_csr_impl(self)->resize_as_(src);
  }
  return self;
}

}} // namespace at::native