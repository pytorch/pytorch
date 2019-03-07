#ifndef CAFFE2_OPERATORS_SWISH_OP_H_
#define CAFFE2_OPERATORS_SWISH_OP_H_

#include "caffe2/operators/elementwise_ops.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
struct SwishFunctor {
  template <typename T>
  bool operator()(const int N, const T* X, T* Y, Context* context) const;
};

template <class Context>
class SwishGradientOp final : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(SwishGradientOp)

  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float, double>>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& X = Input(0);
    const auto& Y = Input(1);
    const auto& dY = Input(2);
    CAFFE_ENFORCE_EQ(X.numel(), Y.numel());
    CAFFE_ENFORCE_EQ(dY.numel(), Y.numel());
    const int N = X.numel();
    auto* dX = Output(0, X.sizes(), at::dtype<T>());
    const T* X_data = X.template data<T>();
    const T* Y_data = Y.template data<T>();
    const T* dY_data = dY.template data<T>();
    T* dX_data = dX->template mutable_data<T>();
    return SwishBackward(N, dY_data, X_data, Y_data, dX_data);
  }

 private:
  template <typename T>
  bool SwishBackward(int N, const T* dY, const T* X, const T* Y, T* dX);
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_SWISH_OP_H_
