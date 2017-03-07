#ifndef CAFFE2_OPERATORS_MATH_OP_H_
#define CAFFE2_OPERATORS_MATH_OP_H_

#include "caffe2/core/common_omp.h"
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"
#include "caffe2/operators/elementwise_op.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

struct PowFunctor {
  explicit PowFunctor(OperatorBase& op) {
    exponent_ = op.GetSingleArgument<float>("exponent", 0);
  }

  template <typename T, class Context>
  inline void
  operator()(const int n, const T* x, T* y, Context* device_context) {
    math::Powx<float, Context>(n, x, exponent_, y, device_context);
  }

  float exponent_;
};
}

#endif
