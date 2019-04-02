#ifndef CAFFE2_OPERATORS_RELU_OP_H_
#define CAFFE2_OPERATORS_RELU_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"
#include "glog/logging.h"

namespace caffe2 {

template <typename dtype, class DeviceContext>
class ReluOp final : public Operator<dtype, DeviceContext> {
 public:
  USE_SIMPLE_CTOR_DTOR(ReluOp);
  USE_OPERATOR_BASE_FUNCTIONS;

  bool RunOnDevice();

 protected:
  INPUT_OUTPUT_STATS(1, 1, 1, 1);
  DISABLE_COPY_AND_ASSIGN(ReluOp);
};

template <typename dtype, class DeviceContext>
class ReluGradientOp final : public Operator<dtype, DeviceContext> {
 public:
  USE_SIMPLE_CTOR_DTOR(ReluGradientOp);
  USE_OPERATOR_BASE_FUNCTIONS;

  bool RunOnDevice();

 protected:
  // Input: X, dY; Output: dX
  INPUT_OUTPUT_STATS(2, 2, 1, 1);
  DISABLE_COPY_AND_ASSIGN(ReluGradientOp);
};

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_RELU_OP_H_
