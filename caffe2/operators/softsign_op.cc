#include "caffe2/operators/elementwise_op.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

struct SoftsignCPUFunctor {
  template <typename T>
  inline void
  operator()(const int n, const T* x, T* y, CPUContext* /*device_context*/) {
    ConstEigenVectorArrayMap<T> x_arr(x, n);
    EigenVectorMap<T>(y, n) = (1 + x_arr.abs()).inverse() * x_arr;
  }
};

struct SoftsignGradientCPUFunctor {
  template <typename T>
  inline void Run(
      const int n,
      const T* x,
      const T* dy,
      T* dx,
      CPUContext* /*device_context*/) {
    ConstEigenVectorArrayMap<T> dy_arr(dy, n);
    ConstEigenVectorArrayMap<T> x_arr(x, n);
    EigenVectorMap<T>(dx, n) = dy_arr * (1 + x_arr.abs()).pow(2).inverse();
  }
};

REGISTER_CPU_OPERATOR(
    Softsign,
    UnaryElementwiseOp<TensorTypes<float>, CPUContext, SoftsignCPUFunctor>);
REGISTER_CPU_OPERATOR(
    SoftsignGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        WithoutBroadcast<SoftsignGradientCPUFunctor>>);

OPERATOR_SCHEMA(Softsign)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Calculates the softsign (x/1+|x|) of the given input tensor element-wise. This
operation can be done in an in-place fashion too, by providing the same input
and output blobs.
)DOC")
    .Input(0, "input", "1-D input tensor")
    .Output(
        0,
        "output",
        "The softsign (x/1+|x|) values of the input tensor "
        "computed element-wise")
    .InheritOnnxSchema("Softsign");

OPERATOR_SCHEMA(SoftsignGradient)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{1, 0}})
    .SetDoc(R"DOC(
Calculates the softsign gradient (sgn(x)/(1+|x|)^2) of the given input tensor
element-wise.
)DOC")
    .Input(0, "input", "1-D input tensor")
    .Input(1, "input", "1-D input tensor")
    .Output(
        0,
        "output",
        "The softsign gradient (sgn(x)/(1+|x|)^2) values of the input tensor "
        "computed element-wise");

class GetSoftsignGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    CAFFE_ENFORCE(
        I(0) != O(0),
        "Cannot compute softsign gradient "
        "if you choose to do an in-place calculation.");

    return SingleGradientDef(
        "SoftsignGradient",
        "",
        vector<string>{I(0), GO(0)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(Softsign, GetSoftsignGradient);

} // namespace caffe2
