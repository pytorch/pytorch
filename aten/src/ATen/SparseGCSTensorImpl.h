#pragma once

#include <ATen/Tensor.h>
#include <ATen/SparseTensorImpl.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/Exception.h>

namespace at {
// TODO: since many methods in SparseTensorImpl can be used by GCS sparse tensor directly
// we probably should have some superclass between TensorImpl and GCSTensorImpl that is
// shared between COO and GCS tensors.
struct CAFFE2_API SparseGCSTensorImpl : public TensorImpl {

  Tensor pointers_;
  Tensor indices_;
  Tensor values_;
  Tensor reduction_;
  Scalar fill_value_;
 public:
  explicit SparseGCSTensorImpl(at::DispatchKeySet, const caffe2::TypeMeta&);
  
  void resize_and_clear_(ArrayRef<int64_t>& size) {
  }

  Tensor pointers() const { std::cout << "in pointers\n"; return pointers_; }
  Tensor indices() const { return indices_; }
  Tensor values() const { return values_; }
  Tensor reduction() const { return reduction_; }

 private :
  
  explicit SparseGCSTensorImpl(at::DispatchKeySet key_set, const caffe2::TypeMeta& data_type,
                               at::Tensor pointers, at::Tensor indices, at::Tensor values, at::Tensor reduction,
                               Scalar fill_value);


};
}
