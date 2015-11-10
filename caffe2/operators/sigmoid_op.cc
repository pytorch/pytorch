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
  inline bool InplaceAllowed() {
    return true;
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
    Sigmoid, UnaryElementwiseOp<float, CPUContext, SigmoidCPUFunctor<float> >);
REGISTER_CPU_OPERATOR(
    SigmoidGradient, BinaryElementwiseOp<float, CPUContext,
                                     SigmoidGradientCPUFunctor<float> >);

struct GetSigmoidGradient : public GetGradientDefBase {
  static vector<OperatorDef>* Create(const OperatorDef& def) {
    return SingleGradientDef(
        "SigmoidGradient", "",
        vector<string>{O(def, 0), GO(def, 0)},
        vector<string>{GI(def, 0)});
  }
};
REGISTER_GRADIENT(Sigmoid, GetSigmoidGradient);
}  // namespace
}  // namespace caffe2