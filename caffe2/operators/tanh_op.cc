#include <cmath>

#include "caffe2/operators/elementwise_op.h"

namespace caffe2 {

template <typename T>
struct TanhCPUFunctor {
  inline void operator()(const int n, const T* x,
                         T* y, CPUContext* device_context) {
    for (int i = 0; i < n; ++i) {
      y[i] = tanh(x[i]);
    }
  }
};

template <typename T>
struct TanhGradientCPUFunctor {
  inline void operator()(const int n, const T* y, const T* dy,
                         T* dx, CPUContext* device_context) {
    for (int i = 0; i < n; ++i) {
      dx[i] = dy[i] * (1 - y[i] * y[i]);
    }
  }
};

namespace {
REGISTER_CPU_OPERATOR(
    Tanh, UnaryElementwiseOp<float, CPUContext, TanhCPUFunctor<float> >);
REGISTER_CPU_OPERATOR(
    TanhGradient, BinaryElementwiseOp<float, CPUContext,
                                     TanhGradientCPUFunctor<float> >);

OPERATOR_SCHEMA(Tanh).NumInputs(1).NumOutputs(1).AllowInplace({{0, 0}});
OPERATOR_SCHEMA(TanhGradient).NumInputs(2).NumOutputs(1).AllowInplace({{1, 0}});

class GetTanhGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "TanhGradient", "",
        std::vector<string>{O(0), GO(0)},
        std::vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(Tanh, GetTanhGradient);
}  // namespace
}  // namespace caffe2
