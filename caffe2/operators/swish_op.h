#pragma once

#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {
template <class Context>
class SwishGradientOp final : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(SwishGradientOp)
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <typename T>
  bool DoRunWithType();

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float, double>>::call(this, Input(X));
  }

 protected:
  INPUT_TAGS(X, Y, DY);
  OUTPUT_TAGS(DX);
};

class GetSwishGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "SwishGradient",
        "",
        vector<string>{I(0), O(0), GO(0)},
        vector<string>{GI(0)});
  }
};

} // namespace caffe2
