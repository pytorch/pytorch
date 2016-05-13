#include <cmath>

#include "caffe2/operators/elementwise_op.h"

namespace caffe2 {

template <typename T>
struct SigmoidCPUFunctor {
  inline void operator()(const int n, const T* x,
                         T* y, CPUContext* device_context) {
    for (int i = 0; i < n; ++i) {
      y[i] = 1. / (1. + exp(-x[i]));
    }
  }
};

template <typename T>
struct SigmoidGradientCPUFunctor {
  inline void operator()(const int n, const T* y, const T* dy,
                         T* dx, CPUContext* device_context) {
    for (int i = 0; i < n; ++i) {
      dx[i] = dy[i] * y[i] * (1. - y[i]);
    }
  }
};

namespace {
REGISTER_CPU_OPERATOR(
    Sigmoid, UnaryElementwiseOp<float, CPUContext, SigmoidCPUFunctor<float> >);
REGISTER_CPU_OPERATOR(
    SigmoidGradient, BinaryElementwiseOp<float, CPUContext,
                                     SigmoidGradientCPUFunctor<float> >);

// Input: X, output: Y
OPERATOR_SCHEMA(Sigmoid).NumInputs(1).NumOutputs(1).AllowInplace({{0, 0}});
// Input: Y, dY, output: dX
OPERATOR_SCHEMA(SigmoidGradient)
    .NumInputs(2).NumOutputs(1).AllowInplace({{1, 0}});

class GetSigmoidGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "SigmoidGradient", "",
        vector<string>{O(0), GO(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(Sigmoid, GetSigmoidGradient);
}  // namespace
}  // namespace caffe2
