// Basic functions on sparse tensors

#include <ATen/ATen.h>
#include <ATen/Layout.h>
#include <ATen/Parallel.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/SparseGCSTensorImpl.h>
#include <ATen/NativeFunctions.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/SparseTensorUtils.h>

#include <TH/THBlasUtils.h>

namespace at { namespace native {

using namespace at::sparse;

// Construction of GCS tensors.
SparseTensor new_gcs_tensor(const TensorOptions& options) {
  // TODO: remove this comment after enabling autograd support for GCS tensor constructor.
  // TORCH_INTERNAL_ASSERT(impl::variable_excluded_from_dispatch());
  AT_ASSERT(options.layout() == kSparseGCS);
  DispatchKey dispatch_key;
  if (options.device().is_cuda()) {
    dispatch_key = DispatchKey::SparseGCS_CUDA;
  } else {
    dispatch_key = DispatchKey::SparseGCS_CPU;
  }
  
  return detail::make_tensor<SparseGCSTensorImpl>(
                                                  DispatchKeySet(dispatch_key), options.dtype());
}

// TODO: This constructor should probably use an ATen abstract method in order to make
// autograd dispatch available for the GCS constructor. See the relevant note in native_functions.yaml. 
Tensor sparse_gcs_tensor(const Tensor& pointers, const Tensor& indices, const Tensor& values,
                         const Tensor& reduction, IntArrayRef size,
                         const TensorOptions& options) {
  TORCH_CHECK(!options.has_layout() || options.layout() == kSparseGCS, "expected sparse GCS layout, but got layout ", options.layout());
  
  SparseTensor self = new_gcs_tensor(options);
  int64_t nnz_size = values.numel();
  int64_t ptr_size = pointers.numel();
  int64_t redux_size = reduction.numel();
  
  get_sparse_impl<SparseGCSTensorImpl>(self)->resize_and_clear_(nnz_size, ptr_size, redux_size, size);
  get_sparse_impl<SparseGCSTensorImpl>(self)->set_member_tensors_unsafe(pointers,
                                                                        indices, values, reduction);
    
  return self;
}

// Access members of GCS tensors.
int64_t _nnz_sparse_gcs(const SparseTensor& self) {
  return get_sparse_impl<SparseGCSTensorImpl>(self)->nnz();
}

Tensor values_sparse_gcs(const Tensor& self) {
  return get_sparse_impl<SparseGCSTensorImpl>(self)->values().alias();      
}

Tensor pointers_sparse_gcs(const Tensor& self) {
  return get_sparse_impl<SparseGCSTensorImpl>(self)->pointers().alias();      
}

Tensor indices_sparse_gcs(const Tensor& self) {
  return get_sparse_impl<SparseGCSTensorImpl>(self)->indices().alias();      
}

Tensor reduction_sparse_gcs(const Tensor& self) {
  return get_sparse_impl<SparseGCSTensorImpl>(self)->reduction().alias();      
}

}} // namespace at::native