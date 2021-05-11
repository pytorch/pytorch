#include "caffe2/operators/sinh_op.h"

#include <algorithm>
#include <functional>

namespace caffe2 {

template <>
template <typename T>
bool SinhGradientFunctor<CPUContext>::Forward(
    const std::vector<int>& /* dY_dims */,
    const std::vector<int>& X_dims,
    const T* dY,
    const T* X,
    T* dX,
    CPUContext* /* context */) const {
  const int size = std::accumulate(
      // NOLINTNEXTLINE(modernize-use-transparent-functors)
      X_dims.cbegin(), X_dims.cend(), 1, std::multiplies<int>());
  ConstEigenVectorArrayMap<T> dY_arr(dY, size);
  ConstEigenVectorArrayMap<T> X_arr(X, size);
  EigenVectorMap<T>(dX, size) = dY_arr * (X_arr.exp() + (-X_arr).exp()) / 2;
  return true;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    Sinh,
    UnaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        SinhFunctor<CPUContext>>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    SinhGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        SinhGradientFunctor<CPUContext>>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(Sinh)
    .NumInputs(1)
    .NumOutputs(1)
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Calculates the hyperbolic sine of the given input tensor, element-wise.

Github Links:

- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/sinh_op.cc


<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "Sinh",
    ["X"],
    ["Y"]
)

workspace.FeedBlob("X", np.random.rand(5).astype(np.float32))
print("X:", workspace.FetchBlob("X"))
workspace.RunOperatorOnce(op)
print("Y:", workspace.FetchBlob("Y"))

```

**Result**

```

X: [0.98907769 0.52907848 0.03216429 0.94983935 0.47881418]
Y: [1.15841695 0.5541099  0.03216984 1.09924557 0.49732079]

```

</details>

)DOC")
    .Input(0, "input", "Input tensor")
    .Output(
        0,
        "output",
        "The hyperbolic sine values of the input tensor, computed "
        "element-wise")
    .InheritOnnxSchema();

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(SinhGradient)
    .NumInputs(2)
    .NumOutputs(1)
    .IdenticalTypeAndShapeOfInput(0);

namespace {

class GetSinhGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  std::vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "SinhGradient",
        "",
        std::vector<std::string>{GO(0), I(0)},
        std::vector<std::string>{GI(0)});
  }
};

} // namespace

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_GRADIENT(Sinh, GetSinhGradient);

} // namespace caffe2
