#ifndef CAFFE2_OPERATORS_THRESHOLDED_RELU_OP_H_
#define CAFFE2_OPERATORS_THRESHOLDED_RELU_OP_H_

#include "caffe2/core/common_omp.h"
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class ThresholdedReluOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  ThresholdedReluOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {
    alpha_ = OperatorBase::GetSingleArgument<T>("alpha", 1.0);
  }

  bool RunOnDevice() override;

 protected:
  T alpha_;
};

template <typename T, class Context>
class ThresholdedReluGradientOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  ThresholdedReluGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {
    alpha_ = OperatorBase::GetSingleArgument<T>("alpha", 1.0);
  }

  bool RunOnDevice() override;

 protected:
  T alpha_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_THRESHOLDED_RELU_OP_H_
