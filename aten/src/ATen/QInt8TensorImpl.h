#pragma once

#include <ATen/Tensor.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/Exception.h>

namespace at {

struct CAFFE2_API QInt8TensorImpl : public TensorImpl {

  float scale_ = 1.0;
  int32_t zero_point_ = 0;
  // thoughts on real value: we dont need another copy of real value to save
  // some memory

public:
  explicit QInt8TensorImpl(Storage&& storage, at::TensorTypeId);
  explicit QInt8TensorImpl(Storage&& storage, at::TensorTypeId, float scale, int32_t zero_point);

  float scale() const { return scale_; }
  void set_scale(float scale) { scale_ = scale; }
  int32_t zero_point() const { return zero_point_; }
  void set_zero_point(int zero_point) { zero_point_ = zero_point; }


};

} // namespace c10
