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

  // Data for making index conversion operations faster.

  // strides of the first half of the split dimensions.
  std::vector<int64_t> strides0_;
  std::vector<int64_t> strides1_, dims0_, dims1_;
  // Dimension at which we split the tensor dimensions into two groups for reduction to
  // a 2D GCS tensor.
  int64_t rsplit_dim_;           
 public:
  explicit SparseGCSTensorImpl(at::DispatchKeySet, const caffe2::TypeMeta&);
  
  void resize_and_clear_(int64_t nnz_size, int64_t ptr_size, int64_t redux_size, ArrayRef<int64_t> size);

  void set_member_tensors_unsafe(const Tensor& pointers, const Tensor& indices, const Tensor& values, const Tensor& reduction,
                                 const Scalar& fill_value);

  Tensor pointers() const { return pointers_; }
  Tensor indices() const { return indices_; }
  Tensor values() const { return values_; }
  Tensor reduction() const { return reduction_; }
  int64_t nnz() const { return values_.size(0); } // TODO: methods like these also exist in COO tensor. Deduplicate?

 private :
  
  explicit SparseGCSTensorImpl(at::DispatchKeySet key_set, const caffe2::TypeMeta& data_type,
                               at::Tensor pointers, at::Tensor indices, at::Tensor values, at::Tensor reduction,
                               Scalar fill_value);

  template <typename T>
  void make_strides(T& shape, std::vector<int64_t>& strides, std::vector<int64_t>& dims);
};
}
