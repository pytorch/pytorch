#ifndef CAFFE2_OPERATORS_SOFTMAX_OP_H_
#define CAFFE2_OPERATORS_SOFTMAX_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"
#include "caffe2/core/logging.h"

namespace caffe2 {

template <typename T, class Context>
class SoftmaxOp final : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(SoftmaxOp);
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;

 protected:
  Tensor<Context> scale_;
  Tensor<Context> sum_multiplier_;
  DISABLE_COPY_AND_ASSIGN(SoftmaxOp);
};

template <typename T, class Context>
class SoftmaxGradientOp final : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(SoftmaxGradientOp);
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;

 protected:
  Tensor<Context> scale_;
  Tensor<Context> sum_multiplier_;
  DISABLE_COPY_AND_ASSIGN(SoftmaxGradientOp);
};

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_SOFTMAX_OP_H_
