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
  inline bool InplaceAllowed() {
    return true;
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
  inline bool InplaceAllowed(const int input_id, const int output_id) {
    if (input_id == 1 && output_id == 0) {
      return true;
    } else {
      return false;
    }
  }
};

namespace {
REGISTER_CPU_OPERATOR(
    Tanh, UnaryElementwiseOp<float, CPUContext, TanhCPUFunctor<float> >);
REGISTER_CPU_OPERATOR(
    TanhGradient, BinaryElementwiseOp<float, CPUContext,
                                     TanhGradientCPUFunctor<float> >);

struct GetTanhGradient : public GetGradientDefBase {
  static vector<OperatorDef>* Create(const OperatorDef& def) {
    return new vector<OperatorDef>{
        CreateOperatorDef(
            "TanhGradient", "",
            std::vector<string>{def.output(0),
                                GradientName(def.output(0))},
            std::vector<string>{GradientName(def.input(0))})};
  }
};
REGISTER_GRADIENT(Tanh, GetTanhGradient);
}  // namespace
}  // namespace caffe2