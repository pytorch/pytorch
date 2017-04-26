#ifndef CAFFE2_OPERATORS_SCALE_GRADIENT_H_
#define CAFFE2_OPERATORS_SCALE_GRADIENT_H_

#include "caffe2/core/operator.h"

namespace caffe2 {

template <class Context>
class ScaleGradientOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  ScaleGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        scale_(OperatorBase::GetSingleArgument<float>("scale", 1.0f)) {}

  bool RunOnDevice() override {
    const auto& in = Input(0);
    auto* out = Output(0);
    if (out != &in) {
      out->CopyFrom(in, &context_);
    }
    return true;
  }

 private:
  float scale_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_SCALE_GRADIENT_H_
