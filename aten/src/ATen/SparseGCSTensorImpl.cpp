#include <ATen/ATen.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/core/LegacyTypeDispatch.h>

namespace at {

SparseGCSTensorImpl::SparseGCSTensorImpl(at::DispatchKeySet key_set, const caffe2::TypeMeta& data_type) {
  hello
}

SparseGCSTensorImpl::SparseGCSTensorImpl(at::DispatchKeySet key_set, const caffe2::TypeMeta& data_type,
                                   at::Tensor pointers, at::Tensor indices, at::Tensor values) {}
}
