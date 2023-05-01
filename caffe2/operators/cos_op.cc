#include "caffe2/operators/cos_op.h"
#include "caffe2/utils/eigen_utils.h"

#include <algorithm>
#include <functional>

namespace caffe2 {

template <>
template <typename T>
bool CosGradientFunctor<CPUContext>::Forward(
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
  EigenVectorMap<T>(dX, size) = -dY_arr * X_arr.sin();
  return true;
}

REGISTER_CPU_OPERATOR(
    Cos,
    UnaryElementwiseOp<TensorTypes<float>, CPUContext, CosFunctor<CPUContext>>);
REGISTER_CPU_OPERATOR(
    CosGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        CosGradientFunctor<CPUContext>>);

OPERATOR_SCHEMA(Cos)
    .NumInputs(1)
    .NumOutputs(1)
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Calculates the cosine of the given input tensor, element-wise.

Github Links:

- https://github.com/pytorch/pytorch/blob/main/caffe2/operators/cos_op.cc


<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "Cos",
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

X: [0.6816719  0.76771533 0.933932   0.01404487 0.11862425]
Y: [0.7765203  0.71949923 0.5946774  0.99990135 0.9929724 ]

```

</details>


)DOC")
    .Input(0, "X", "*(type: Tensor`<float>`)* Input tensor.")
    .Output(
        0,
        "Y",
        "*(type: Tensor`<float>`)* Output tensor calculated as the cosine of the input tensor, element-wise.");

OPERATOR_SCHEMA(CosGradient).NumInputs(2).NumOutputs(1).IdenticalTypeAndShape();

namespace {

class GetCosGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  std::vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "CosGradient",
        "",
        std::vector<std::string>{I(0), GO(0)},
        std::vector<std::string>{GI(0)});
  }
};

} // namespace

REGISTER_GRADIENT(Cos, GetCosGradient);

} // namespace caffe2
