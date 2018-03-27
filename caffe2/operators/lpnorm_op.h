#ifndef CAFFE2_OPERATORS_LPNORM_OP_H_
#define CAFFE2_OPERATORS_LPNORM_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class LpNormOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  LpNormOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        p_(OperatorBase::GetSingleArgument<int>("p", 2)),
        average_(OperatorBase::GetSingleArgument<bool>("average", false)) {
    CAFFE_ENFORCE(p_ == 1 || p_ == 2, "p should be either 1 or 2.");
  }

  bool RunOnDevice() override;

 protected:
  int p_;
  bool average_;
  INPUT_TAGS(X_IN);
  OUTPUT_TAGS(OUT);
  // Input: X; Output: Norm
};

template <typename T, class Context>
class LpNormGradientOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  LpNormGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        p_(OperatorBase::GetSingleArgument<int>("p", 2)),
        average_(OperatorBase::GetSingleArgument<bool>("average", false)) {
    CAFFE_ENFORCE(p_ == 1 || p_ == 2, "p should be either 1 or 2.");
  }

  bool RunOnDevice() override;

 protected:
  int p_;
  bool average_;
  INPUT_TAGS(X_IN, DER_NORM_IN);
  OUTPUT_TAGS(DER_X_OUT);
  // Input: X, dNorm; Output: dX
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_LPNORM_OP_H_
