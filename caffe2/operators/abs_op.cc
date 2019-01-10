#include "caffe2/operators/elementwise_op.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

struct AbsCPUFunctor {
  template <typename T>
  inline void
  operator()(const int n, const T* x, T* y, CPUContext* device_context) {
    math::Abs<T, CPUContext>(n, x, y, device_context);
  }
};

struct AbsGradientCPUFunctor {
  template <typename T>
  inline void
  Run(const int n, const T* x, const T* dy, T* dx, CPUContext* /* unused */) {
    ConstEigenVectorArrayMap<T> dyM(dy, n);
    ConstEigenVectorArrayMap<T> xM(x, n);
    EigenVectorMap<T>(dx, n) =
        (xM == T(0)).select(T(0), (xM > T(0)).select(dyM, -dyM));
  }
};

REGISTER_CPU_OPERATOR(
    Abs,
    UnaryElementwiseOp<TensorTypes<float>, CPUContext, AbsCPUFunctor>);
REGISTER_CPU_OPERATOR(
    AbsGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        WithoutBroadcast<AbsGradientCPUFunctor>>);

OPERATOR_SCHEMA(Abs)
    .NumInputs(1)
    .NumOutputs(1)
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Calculates the absolute value of the given input tensor, element-wise.
)DOC")
    .Input(0, "input", "Input tensor")
    .Output(
        0,
        "output",
        "The absolute value of the input tensor computed element-wise")
    .InheritOnnxSchema("Abs");

OPERATOR_SCHEMA(AbsGradient).NumInputs(2).NumOutputs(1).IdenticalTypeAndShape();

class GetAbsGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "AbsGradient",
        "",
        std::vector<string>{I(0), GO(0)},
        std::vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(Abs, GetAbsGradient);
} // namespace caffe2
