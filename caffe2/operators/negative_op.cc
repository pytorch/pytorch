#include "caffe2/operators/elementwise_op.h"

namespace caffe2 {

struct NegativeCPUFunctor {
  template <typename T>
  inline void
  operator()(const int n, const T* x, T* y, CPUContext* /*device_context*/) {
    EigenVectorMap<T>(y, n) = -ConstEigenVectorMap<T>(x, n);
    // for (int i = 0; i < n; ++i) {
    //  y[i] = -x[i];
    //}
  }
};

REGISTER_CPU_OPERATOR(
    Negative, UnaryElementwiseOp<
        TensorTypes<float, double, int, long>, CPUContext, NegativeCPUFunctor>);

// Input: X, output: Y
OPERATOR_SCHEMA(Negative)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Computes the element-wise negative of the input.
)DOC")
    .Input(0, "X", "1D input tensor")
    .Output(0, "Y", "1D input tensor")
    .InheritOnnxSchema("Neg");

class GetNegativeGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "Negative", "",
        vector<string>{GO(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(Negative, GetNegativeGradient);
}  // namespace caffe2
