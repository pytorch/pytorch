#include <Eigen/Core>
#include "caffe2/operators/elementwise_op.h"

namespace caffe2 {

struct SqrtCPUFunctor {
  template <typename T>
  inline void
  operator()(const int n, const T* x, T* y, CPUContext* /*device_context*/) {
    EigenVectorArrayMap<T>(y, n) = ConstEigenVectorArrayMap<T>(x, n).sqrt();
  }
};

REGISTER_CPU_OPERATOR(
    Sqrt,
    UnaryElementwiseOp<TensorTypes<float>, CPUContext, SqrtCPUFunctor>);
// Input: X, output: Y
OPERATOR_SCHEMA(Sqrt)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Computes the element-wise sqrt of the input.
)DOC")
    .Input(0, "X", "ND input tensor")
    .Output(0, "Y", "ND input tensor");

class GetSqrtGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    Argument scale_arg;
    scale_arg.set_name("scale");
    scale_arg.set_f(0.5);
    return vector<OperatorDef>{CreateOperatorDef(
                                   "Scale",
                                   "",
                                   std::vector<string>{GO(0)},
                                   std::vector<string>{GI(0)},
                                   std::vector<Argument>{scale_arg}),
                               CreateOperatorDef(
                                   "Div",
                                   "",
                                   std::vector<string>{GI(0), O(0)},
                                   std::vector<string>{GI(0)})};
  }
};
REGISTER_GRADIENT(Sqrt, GetSqrtGradient);
} // namespace caffe2
