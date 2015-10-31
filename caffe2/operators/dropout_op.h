#ifndef CAFFE2_OPERATORS_DROPOUT_OP_H_
#define CAFFE2_OPERATORS_DROPOUT_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"
#include "caffe2/core/logging.h"

namespace caffe2 {

template <typename T, class Context>
class DropoutOp final : public Operator<Context> {
 public:
  USE_OPERATOR_BASE_FUNCTIONS;
  DropoutOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        ratio_(OperatorBase::GetSingleArgument<float>("ratio", 0.5)) {
    CAFFE_DCHECK_GE(ratio_, 0);
    CAFFE_DCHECK_LT(ratio_, 1);
  }

  bool RunOnDevice() override;

 protected:
  float ratio_;
  // Input: X; Output: Y, mask.
  INPUT_OUTPUT_STATS(1, 1, 2, 2);
  DISABLE_COPY_AND_ASSIGN(DropoutOp);
};

template <typename T, class Context>
class DropoutGradientOp final : public Operator<Context> {
 public:
  USE_OPERATOR_BASE_FUNCTIONS;
  DropoutGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        ratio_(OperatorBase::GetSingleArgument<float>("ratio", 0.5)) {
    CAFFE_DCHECK_GE(ratio_, 0);
    CAFFE_DCHECK_LT(ratio_, 1);
  }

  bool RunOnDevice() override;

 protected:
  float ratio_;
  // Input: dY, mask; Output: dX
  INPUT_OUTPUT_STATS(2, 2, 1, 1);
  DISABLE_COPY_AND_ASSIGN(DropoutGradientOp);
};

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_DROPOUT_OP_H_
