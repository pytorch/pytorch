// Basic functions on sparse tensors

#include <ATen/ATen.h>
#include <ATen/Layout.h>
#include <ATen/Parallel.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/CompressedSparseTensorImpl.h>
#include <ATen/NativeFunctions.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/CompressedSparseTensorUtils.h>

#include <TH/THBlasUtils.h>

namespace at { namespace native {

using namespace at::sparse;

// Construction of GCS tensors.
SparseTensor new_gcs_tensor(const TensorOptions& options) {
  // TODO: remove this comment after enabling autograd support for GCS tensor constructor.
  // TORCH_INTERNAL_ASSERT(impl::variable_excluded_from_dispatch());
  AT_ASSERT(options.layout() == kCompressedSparse);
  DispatchKey dispatch_key;
  if (options.device().is_cuda()) {
    dispatch_key = DispatchKey::CompressedSparseCUDA;
  } else {
    dispatch_key = DispatchKey::CompressedSparseCPU;
  }
  
  return detail::make_tensor<CompressedSparseTensorImpl>(
                                                  DispatchKeySet(dispatch_key), options.dtype());
}

// TODO: This constructor should probably use an ATen abstract method in order to make
// autograd dispatch available for the GCS constructor. See the relevant note in native_functions.yaml. 
Tensor sparse_csr_tensor(const Tensor& crow_indices, const Tensor& col_indices, 
                         const Tensor& values,
                         const Tensor& reduction, IntArrayRef size,
                         const TensorOptions& options) {
  TORCH_CHECK(!options.has_layout() || options.layout() == kCompressedSparse, "expected sparse GCS layout, but got layout ", options.layout());
  
  SparseTensor self = new_gcs_tensor(options);
  int64_t nnz_size = values.numel();
  int64_t ptr_size = crow_indices.numel();
  int64_t redux_size = reduction.numel();
  
  get_sparse_gcs_impl(self)->resize_and_clear_(nnz_size, ptr_size, redux_size, size);
  get_sparse_gcs_impl(self)->set_member_tensors_unsafe(crow_indices, col_indices, values, reduction);
  
  return self;
}

// Access members of GCS tensors.
int64_t _nnz_sparse_gcs(const SparseTensor& self) {
  return get_sparse_gcs_impl(self)->nnz();
}

Tensor values_sparse_gcs(const Tensor& self) {
  return get_sparse_gcs_impl(self)->values().alias();      
}

Tensor crow_indices_sparse_gcs(const Tensor& self) {
  return get_sparse_gcs_impl(self)->crow_indices().alias();      
}

Tensor col_indices_sparse_gcs(const Tensor& self) {
  return get_sparse_gcs_impl(self)->col_indices().alias();      
}

Tensor reduction_sparse_gcs(const Tensor& self) {
  return get_sparse_gcs_impl(self)->reduction().alias();      
}

bool _is_same_size_as_sparse_gcs(const SparseTensor& self, const SparseTensor& src) {
  return self.dim() == src.dim() && self.sizes().equals(src.sizes());
}

SparseTensor& resize_as_sparse_gcs_(SparseTensor& self, const SparseTensor& src) {
  if (!_is_same_size_as_sparse_gcs(self, src)) {
    get_sparse_gcs_impl(self)->resize_as_(src);
  }
  return self;
}

}} // namespace at::native