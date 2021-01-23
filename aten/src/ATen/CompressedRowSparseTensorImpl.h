#pragma once

#include <ATen/Tensor.h>
#include <ATen/TensorUtils.h>
#include <ATen/SparseTensorUtils.h>
#include <ATen/SparseTensorImpl.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/Exception.h>

namespace at {

struct TORCH_API CompressedRowSparseTensorImpl : public TensorImpl {
  Tensor crow_indices_;
  Tensor col_indices_;
  Tensor values_;

 public:
  explicit CompressedRowSparseTensorImpl(at::DispatchKeySet, const caffe2::TypeMeta&);

  void resize_and_clear_(int64_t nnz_size, IntArrayRef size);
  void resize_as_(const Tensor& src);
  void set_member_tensors_unsafe(const Tensor& crow_indices, const Tensor& col_indices,
                                 const Tensor& values);
  
  Tensor crow_indices() const { return crow_indices_; }
  Tensor col_indices() const { return col_indices_; }
  Tensor values() const { return values_; }
  int nnz() const { return values_.size(0); }

 private :
  
  explicit CompressedRowSparseTensorImpl(at::DispatchKeySet key_set, const caffe2::TypeMeta& data_type,
                               at::Tensor crow_indices, at::Tensor col_indices, at::Tensor values);
};
} // namespace at
