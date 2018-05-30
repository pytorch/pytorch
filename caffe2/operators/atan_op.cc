#include "caffe2/operators/elementwise_op.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

struct AtanCPUFunctor {
  template <typename T>
  inline void
  operator()(const int n, const T* x, T* y, CPUContext* device_context) {
    math::Atan<T, CPUContext>(n, x, y, device_context);
  }
};

struct AtanGradientCPUFunctor {
  template <typename T>
  inline void
  Run(const int n, const T* x, const T* dy, T* dx, CPUContext* /* unused */) {
    ConstEigenVectorArrayMap<T> dyM(dy, n);
    ConstEigenVectorArrayMap<T> xM(x, n);
    EigenVectorMap<T>(dx, n) = dyM / (1 + xM * xM);
  }
};

REGISTER_CPU_OPERATOR(
    Atan,
    UnaryElementwiseOp<TensorTypes<float>, CPUContext, AtanCPUFunctor>);
REGISTER_CPU_OPERATOR(
    AtanGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        WithoutBroadcast<AtanGradientCPUFunctor>>);

OPERATOR_SCHEMA(Atan)
    .NumInputs(1)
    .NumOutputs(1)
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Calculates the arctangent of the given input tensor, element-wise.
)DOC")
    .Input(0, "input", "Input tensor")
    .Output(
        0,
        "output",
        "The arctangent of the input tensor computed element-wise");

OPERATOR_SCHEMA(AtanGradient).NumInputs(2).NumOutputs(1).IdenticalTypeAndShape();

class GetAtanGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "AtanGradient",
        "",
        std::vector<string>{I(0), GO(0)},
        std::vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(Atan, GetAtanGradient);
} // namespace caffe2
