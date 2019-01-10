#include <cmath>

#include "caffe2/operators/elementwise_op.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

struct TanhCPUFunctor {
  template <typename T>
  inline void
  operator()(const int n, const T* x, T* y, CPUContext* /*device_context*/) {
#ifdef CAFFE2_USE_ACCELERATE
    vvtanhf(y, x, &n);
#else
    ConstEigenVectorArrayMap<T> x_arr(x, n);
    EigenVectorMap<T>(y, n) = 1 - 2 * ((x_arr * 2).exp() + 1).inverse();
#endif
  }
};

struct TanhGradientCPUFunctor {
  template <typename T>
  inline void Run(
      const int n,
      const T* y,
      const T* dy,
      T* dx,
      CPUContext* /*device_context*/) {
    ConstEigenVectorArrayMap<T> dy_arr(dy, n);
    ConstEigenVectorArrayMap<T> y_arr(y, n);
    EigenVectorMap<T>(dx, n) = dy_arr * (1 - y_arr * y_arr);
  }
};

REGISTER_CPU_OPERATOR(
    Tanh, UnaryElementwiseOp<TensorTypes<float>, CPUContext, TanhCPUFunctor>);
REGISTER_CPU_OPERATOR(
    TanhGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        WithoutBroadcast<TanhGradientCPUFunctor>>);

OPERATOR_SCHEMA(Tanh)
  .NumInputs(1)
  .NumOutputs(1)
  .AllowInplace({{0, 0}})
  .IdenticalTypeAndShape()
  .SetDoc(R"DOC(
Calculates the hyperbolic tangent of the given input tensor element-wise. This
operation can be done in an in-place fashion too, by providing the same input
and output blobs.
)DOC")
  .Input(0, "input", "1-D input tensor")
  .Output(0, "output", "The hyperbolic tangent values of the input tensor "
          "computed element-wise")
  .InheritOnnxSchema("Tanh");

OPERATOR_SCHEMA(TanhGradient).NumInputs(2).NumOutputs(1).AllowInplace({{1, 0}});

class GetTanhGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "TanhGradient", "",
        std::vector<string>{O(0), GO(0)},
        std::vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(Tanh, GetTanhGradient);
}  // namespace caffe2
