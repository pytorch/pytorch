#include "caffe2/operators/elementwise_op.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

struct TanCPUFunctor {
  template <typename T>
  inline void
  operator()(const int n, const T* x, T* y, CPUContext* device_context) {
    math::Tan<T, CPUContext>(n, x, y, device_context);
  }
};

struct TanGradientCPUFunctor {
  template <typename T>
  inline void
  Run(const int n, const T* x, const T* dy, T* dx, CPUContext* /* unused */) {
    ConstEigenVectorArrayMap<T> dyM(dy, n);
    ConstEigenVectorArrayMap<T> xM(x, n);
    EigenVectorMap<T>(dx, n) = dyM / square(cos(xM));
  }
};

REGISTER_CPU_OPERATOR(
    Tan,
    UnaryElementwiseOp<TensorTypes<float>, CPUContext, TanCPUFunctor>);
REGISTER_CPU_OPERATOR(
    TanGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        WithoutBroadcast<TanGradientCPUFunctor>>);

OPERATOR_SCHEMA(Tan)
    .NumInputs(1)
    .NumOutputs(1)
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Calculates the tangent of the given input tensor, element-wise.
)DOC")
    .Input(0, "input", "Input tensor")
    .Output(
        0,
        "output",
        "The tangent of the input tensor computed element-wise");

OPERATOR_SCHEMA(TanGradient).NumInputs(2).NumOutputs(1).IdenticalTypeAndShape();

class GetTanGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "TanGradient",
        "",
        std::vector<string>{I(0), GO(0)},
        std::vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(Tan, GetTanGradient);
} // namespace caffe2
