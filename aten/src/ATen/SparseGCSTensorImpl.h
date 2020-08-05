#pragma once

#include <ATen/Tensor.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/Exception.h>

namespace at {
struct CAFFE2_API SparseGCSTensorImpl : public TensorImpl {
 public:
  explicit SparseGCSTensorImpl(at::DispatchKeySet, const caffe2::TypeMeta&);

  void resize_and_clear_(ArrayRef<int64_t>& size) {
  }
};
}
