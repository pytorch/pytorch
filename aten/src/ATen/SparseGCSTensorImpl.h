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
  
  void resize_and_clear_(int nnz_size, int ptr_size, int redux_size) {
    // TODO: perform error checking.

    // call pointers().options() here since the struct contructor calls the tensor constructor
    // with args for device specific init.
    auto empty_pointers = at::empty(ptr_size, pointers().options());
    auto empty_indices = at::empty(nnz_size, indices().options());
    auto empty_values = at::empty(nnz_size, values().options());
    auto empty_reduction = at::empty(redux_size, reduction().options());

    // directly set to the member variables. there should be lots of error checking here.
    pointers_ = empty_pointers;
    indices_ = empty_indices;
    values_ = empty_values;
    reduction_ = empty_reduction;
  }

  Tensor pointers() const { return pointers_; }
  Tensor indices() const { return indices_; }
  Tensor values() const { return values_; }
  Tensor reduction() const { return reduction_; }

 private :
  
  explicit SparseGCSTensorImpl(at::DispatchKeySet key_set, const caffe2::TypeMeta& data_type,
                               at::Tensor pointers, at::Tensor indices, at::Tensor values, at::Tensor reduction,
                               Scalar fill_value);


};
}
