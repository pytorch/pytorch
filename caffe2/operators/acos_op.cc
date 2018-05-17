#include "caffe2/operators/elementwise_op.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

struct AcosCPUFunctor {
  template <typename T>
  inline void
  operator()(const int n, const T* x, T* y, CPUContext* device_context) {
    math::Acos<T, CPUContext>(n, x, y, device_context);
  }
};

struct AcosGradientCPUFunctor {
  template <typename T>
  inline void
  Run(const int n, const T* x, const T* dy, T* dx, CPUContext* /* unused */) {
    ConstEigenVectorArrayMap<T> dyM(dy, n);
    ConstEigenVectorArrayMap<T> xM(x, n);
    EigenVectorMap<T>(dx, n) = -dyM / sqrt(1 - xM * xM);
  }
};

REGISTER_CPU_OPERATOR(
    Acos,
    UnaryElementwiseOp<TensorTypes<float>, CPUContext, AcosCPUFunctor>);
REGISTER_CPU_OPERATOR(
    AcosGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        WithoutBroadcast<AcosGradientCPUFunctor>>);

OPERATOR_SCHEMA(Acos)
    .NumInputs(1)
    .NumOutputs(1)
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Calculates the arccosine of the given input tensor, element-wise.
)DOC")
    .Input(0, "input", "Input tensor")
    .Output(
        0,
        "output",
        "The arccosine of the input tensor computed element-wise");

OPERATOR_SCHEMA(AcosGradient).NumInputs(2).NumOutputs(1).IdenticalTypeAndShape();

class GetAcosGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "AcosGradient",
        "",
        std::vector<string>{I(0), GO(0)},
        std::vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(Acos, GetAcosGradient);
} // namespace caffe2
