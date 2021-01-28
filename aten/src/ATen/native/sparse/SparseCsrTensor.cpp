// Basic functions on sparse tensors

#include <ATen/ATen.h>
#include <ATen/Layout.h>
#include <ATen/Parallel.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/SparseCsrTensorImpl.h>
#include <ATen/NativeFunctions.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/SparseCsrTensorUtils.h>

#include <TH/THBlasUtils.h>

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
  TORCH_CHECK(crow_indices.numel() == size[0] + 1, 
    "crow_indices.numel() must be size(0)+1, but got: ", crow_indices.numel());

  SparseTensor self = new_csr_tensor(options);
  get_sparse_csr_impl(self)->resize_and_clear_(values.numel(), size);
  get_sparse_csr_impl(self)->set_member_tensors_unsafe(crow_indices, col_indices, values);
  
  return self;
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