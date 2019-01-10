#include "caffe2/operators/elementwise_op.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

struct SigmoidCPUFunctor {
  template <typename T>
  inline void
  operator()(const int n, const T* x, T* y, CPUContext* /*device_context*/) {
    ConstEigenVectorArrayMap<T> xM(x, n);
    EigenVectorArrayMap<T>(y, n) = 1. / (1. + (-xM).exp());
  }
};

struct SigmoidGradientCPUFunctor {
  template <typename T>
  inline void Run(
      const int n,
      const T* y,
      const T* dy,
      T* dx,
      CPUContext* /*device_context*/) {
    ConstEigenVectorArrayMap<T> yM(y, n), dyM(dy, n);
    EigenVectorArrayMap<T>(dx, n) = dyM * yM * (1. - yM);
  }
};

REGISTER_CPU_OPERATOR(
    Sigmoid, UnaryElementwiseOp<
        TensorTypes<float>, CPUContext, SigmoidCPUFunctor>);
REGISTER_CPU_OPERATOR(
    SigmoidGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        WithoutBroadcast<SigmoidGradientCPUFunctor>>);

// Input: X, output: Y
OPERATOR_SCHEMA(Sigmoid)
  .NumInputs(1)
  .NumOutputs(1)
  .AllowInplace({{0, 0}})
  .IdenticalTypeAndShape()
  .SetDoc(R"DOC(
Sigmoid takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the sigmoid function, y = 1 / (1 + exp(-x)), is applied to the
tensor elementwise.
)DOC")
  .Input(0, "X", "1D input tensor")
  .Output(0, "Y", "1D output tensor")
  .InheritOnnxSchema("Sigmoid");
// Input: Y, dY, output: dX
OPERATOR_SCHEMA(SigmoidGradient)
  .NumInputs(2)
  .NumOutputs(1)
  .AllowInplace({{1, 0}})
  .SetDoc(R"DOC(
SigmoidGradient takes both Y and dY and uses this to update dX according to the
chain rule and derivatives of the sigmoid function.
)DOC");

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
}  // namespace caffe2
