#include "caffe2/operators/elementwise_op.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

struct AsinCPUFunctor {
  template <typename T>
  inline void
  operator()(const int n, const T* x, T* y, CPUContext* device_context) {
    math::Asin<T, CPUContext>(n, x, y, device_context);
  }
};

struct AsinGradientCPUFunctor {
  template <typename T>
  inline void
  Run(const int n, const T* x, const T* dy, T* dx, CPUContext* /* unused */) {
    ConstEigenVectorArrayMap<T> dyM(dy, n);
    ConstEigenVectorArrayMap<T> xM(x, n);
    EigenVectorMap<T>(dx, n) = dyM / sqrt(1 - xM * xM);
  }
};

REGISTER_CPU_OPERATOR(
    Asin,
    UnaryElementwiseOp<TensorTypes<float>, CPUContext, AsinCPUFunctor>);
REGISTER_CPU_OPERATOR(
    AsinGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        WithoutBroadcast<AsinGradientCPUFunctor>>);

OPERATOR_SCHEMA(Asin)
    .NumInputs(1)
    .NumOutputs(1)
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Calculates the arcsine of the given input tensor, element-wise.
)DOC")
    .Input(0, "input", "Input tensor")
    .Output(
        0,
        "output",
        "The arcsine of the input tensor computed element-wise");

OPERATOR_SCHEMA(AsinGradient).NumInputs(2).NumOutputs(1).IdenticalTypeAndShape();

class GetAsinGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "AsinGradient",
        "",
        std::vector<string>{I(0), GO(0)},
        std::vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(Asin, GetAsinGradient);
} // namespace caffe2
