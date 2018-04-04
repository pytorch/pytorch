#include "swish_op.h"
#include "caffe2/core/types.h"
#include "caffe2/operators/elementwise_op.h"
#include "caffe2/utils/math.h"

namespace caffe2 {
struct SwishCPUFunctor {
  template <typename T>
  inline void
  operator()(const int n, const T* x, T* y, CPUContext* /*device_context*/) {
    ConstEigenVectorArrayMap<T> xM(x, n);
    EigenVectorArrayMap<T>(y, n) = xM / (1. + (-xM).exp());
  }
};

template <>
template <typename T>
bool SwishGradientOp<CPUContext>::DoRunWithType() {
  auto& Xin = Input(X);
  auto& Yin = Input(Y);
  auto& DYin = Input(DY);
  auto* DXout = Output(DX);
  CAFFE_ENFORCE_EQ(Xin.size(), Yin.size());
  CAFFE_ENFORCE_EQ(DYin.size(), Yin.size());
  DXout->ResizeLike(Yin);

  const float* Xdata = Xin.template data<float>();
  const float* Ydata = Yin.template data<float>();
  const float* dYdata = DYin.template data<float>();
  float* dXdata = DXout->template mutable_data<float>();

  EigenVectorArrayMap<float> dXvec(dXdata, DXout->size());
  ConstEigenVectorArrayMap<float> Xvec(Xdata, Xin.size());
  ConstEigenVectorArrayMap<float> Yvec(Ydata, Yin.size());
  ConstEigenVectorArrayMap<float> dYvec(dYdata, DYin.size());

  // dx = dy * (y + sigmoid(x)*(1-y))
  dXvec = dYvec * (Yvec + (1. / (1. + (-Xvec).exp())) * (1. - Yvec));
  return true;
}

REGISTER_CPU_OPERATOR(
    Swish,
    UnaryElementwiseOp<
        TensorTypes<float, double>,
        CPUContext,
        SwishCPUFunctor>);
REGISTER_CPU_OPERATOR(SwishGradient, SwishGradientOp<CPUContext>);

// Input: X, output: Y
OPERATOR_SCHEMA(Swish)
    .NumInputs(1)
    .NumOutputs(1)
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Swish takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the swish function, y = x / (1 + exp(-x)), is applied to the
tensor elementwise.
)DOC")
    .Input(0, "X", "1D input tensor")
    .Output(0, "Y", "1D output tensor");
// Input: X, Y, dY, output: dX
OPERATOR_SCHEMA(SwishGradient)
    .NumInputs(3)
    .NumOutputs(1)
    .AllowInplace({{2, 0}})
    .SetDoc(R"DOC(
SwishGradient takes X, Y and dY and uses this to update dX according to the
chain rule and derivatives of the swish function.
)DOC");

REGISTER_GRADIENT(Swish, GetSwishGradient);
} // namespace caffe2
