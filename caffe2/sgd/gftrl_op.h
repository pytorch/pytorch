#pragma once

#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T>
struct GFtrlParams {
  explicit GFtrlParams(OperatorBase* op)
      : alphaInv(1.0 / op->GetSingleArgument<float>("alpha", 0.005f)),
        beta(op->GetSingleArgument<float>("beta", 1.0f)),
        lambda1(op->GetSingleArgument<float>("lambda1", 0.001f)),
        lambda2(op->GetSingleArgument<float>("lambda2", 0.001f)) {}
  T alphaInv;
  T beta;
  T lambda1;
  T lambda2;
};

template <typename T, class Context>
class GFtrlOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  GFtrlOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws), params_(this) {
    CAFFE_ENFORCE(
        !HasArgument("alpha") || ALPHA >= InputSize(),
        "Cannot specify alpha by both input and argument");
  }
  bool RunOnDevice() override;

 protected:
  GFtrlParams<T> params_;
  INPUT_TAGS(VAR, N_Z, GRAD, ALPHA);
  OUTPUT_TAGS(OUTPUT_VAR, OUTPUT_N_Z);
};

} // namespace caffe2
