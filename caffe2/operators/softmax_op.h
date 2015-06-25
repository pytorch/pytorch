#ifndef CAFFE2_OPERATORS_SOFTMAX_OP_H_
#define CAFFE2_OPERATORS_SOFTMAX_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"
#include "glog/logging.h"

namespace caffe2 {

template <typename dtype, class DeviceContext>
class SoftmaxOp final : public Operator<dtype, DeviceContext> {
 public:
  USE_SIMPLE_CTOR_DTOR(SoftmaxOp);
  USE_OPERATOR_BASE_FUNCTIONS;
  bool RunOnDevice() override;

 protected:
  Tensor<dtype, DeviceContext> scale_;
  Tensor<dtype, DeviceContext> sum_multiplier_;
  INPUT_OUTPUT_STATS(1, 1, 1, 1);
  DISABLE_COPY_AND_ASSIGN(SoftmaxOp);
};

template <typename dtype, class DeviceContext>
class SoftmaxGradientOp final : public Operator<dtype, DeviceContext> {
 public:
  USE_SIMPLE_CTOR_DTOR(SoftmaxGradientOp);
  USE_OPERATOR_BASE_FUNCTIONS;
  bool RunOnDevice() override;

 protected:
  Tensor<dtype, DeviceContext> scale_;
  Tensor<dtype, DeviceContext> sum_multiplier_;
  // Input: Y, dY. Output: dX
  INPUT_OUTPUT_STATS(2, 2, 1, 1);
  DISABLE_COPY_AND_ASSIGN(SoftmaxGradientOp);
};

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_SOFTMAX_OP_H_
