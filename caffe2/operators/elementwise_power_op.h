#pragma once

#include "caffe2/core/operator.h"
#include "caffe2/operators/elementwise_op.h"

namespace caffe2 {

struct PowCPUFunctor {
  explicit PowCPUFunctor(OperatorBase& op) {
    exponent_ = op.GetSingleArgument<float>("exponent", 0);
  }

  template <typename T>
  inline void
  operator()(const int n, const T* x, T* y, CPUContext* device_context) {
    math::Powx<T, CPUContext>(n, x, exponent_, y, device_context);
  }

  float exponent_;
};

} // namespace caffe2
