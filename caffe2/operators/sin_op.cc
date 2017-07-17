#include "caffe2/operators/elementwise_op.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

struct SinCPUFunctor {
  template <typename T>
  inline void
  operator()(const int n, const T* x, T* y, CPUContext* device_context) {
    math::Sin<T, CPUContext>(n, x, y, device_context);
  }
};

struct SinGradientCPUFunctor {
  template <typename T>
  inline void
  Run(const int n, const T* x, const T* dy, T* dx, CPUContext* /* unused */) {
    ConstEigenVectorArrayMap<T> dyM(dy, n);
    ConstEigenVectorArrayMap<T> xM(x, n);
    EigenVectorMap<T>(dx, n) = dyM * cos(xM);
  }
};

REGISTER_CPU_OPERATOR(
    Sin,
    UnaryElementwiseOp<TensorTypes<float>, CPUContext, SinCPUFunctor>);
REGISTER_CPU_OPERATOR(
    SinGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        WithoutBroadcast<SinGradientCPUFunctor>>);

OPERATOR_SCHEMA(Sin)
    .NumInputs(1)
    .NumOutputs(1)
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Calculates the sine of the given input tensor, element-wise.
)DOC")
    .Input(0, "input", "Input tensor")
    .Output(0, "output", "The sine of the input tensor computed element-wise");

OPERATOR_SCHEMA(SinGradient).NumInputs(2).NumOutputs(1).IdenticalTypeAndShape();

class GetSinGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "SinGradient",
        "",
        std::vector<string>{I(0), GO(0)},
        std::vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(Sin, GetSinGradient);
} // namespace caffe2
