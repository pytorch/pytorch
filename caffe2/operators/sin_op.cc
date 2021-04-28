#include "caffe2/operators/sin_op.h"
#include "caffe2/utils/eigen_utils.h"

#include <algorithm>
#include <functional>

namespace caffe2 {

template <>
template <typename T>
bool SinGradientFunctor<CPUContext>::Forward(
    const std::vector<int>& X_dims,
    const std::vector<int>& /* dY_dims */,
    const T* X,
    const T* dY,
    T* dX,
    CPUContext* /* context */) const {
  const int size = std::accumulate(
      // NOLINTNEXTLINE(modernize-use-transparent-functors)
      X_dims.cbegin(), X_dims.cend(), 1, std::multiplies<int>());
  ConstEigenVectorArrayMap<T> dY_arr(dY, size);
  ConstEigenVectorArrayMap<T> X_arr(X, size);
  EigenVectorMap<T>(dX, size) = dY_arr * X_arr.cos();
  return true;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    Sin,
    UnaryElementwiseOp<TensorTypes<float>, CPUContext, SinFunctor<CPUContext>>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    SinGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        SinGradientFunctor<CPUContext>>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(Sin)
    .NumInputs(1)
    .NumOutputs(1)
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Calculates the sine of the given input tensor, element-wise.

Github Links:

- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/sin_op.cc


<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "Sin",
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

X: [0.8466114  0.1803606  0.5601509  0.04959291 0.64770824]
Y: [0.74903965 0.17938434 0.5313141  0.04957259 0.60336035]

```

</details>

)DOC")
.Input(0, "X", "*(type: Tensor`<float>`)* Input tensor.")
.Output(
    0,
    "Y",
    "*(type: Tensor`<float>`)* Output tensor calculated as the sine of the input tensor, element-wise.");

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(SinGradient).NumInputs(2).NumOutputs(1).IdenticalTypeAndShape();

namespace {

class GetSinGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  std::vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "SinGradient",
        "",
        std::vector<std::string>{I(0), GO(0)},
        std::vector<std::string>{GI(0)});
  }
};

} // namespace

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_GRADIENT(Sin, GetSinGradient);

} // namespace caffe2
