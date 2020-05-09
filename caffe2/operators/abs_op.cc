#include "caffe2/operators/abs_op.h"
#include "caffe2/utils/eigen_utils.h"

#include <algorithm>
#include <functional>

namespace caffe2 {

template <>
template <typename T>
bool AbsGradientFunctor<CPUContext>::Forward(
    const std::vector<int>& X_dims,
    const std::vector<int>& /* dY_dims */,
    const T* X,
    const T* dY,
    T* dX,
    CPUContext* /* context */) const {
  const int size = std::accumulate(
      X_dims.cbegin(), X_dims.cend(), 1, std::multiplies<int>());
  ConstEigenVectorArrayMap<T> dY_arr(dY, size);
  ConstEigenVectorArrayMap<T> X_arr(X, size);
  EigenVectorMap<T>(dX, size) =
      (X_arr == T(0)).select(T(0), (X_arr > T(0)).select(dY_arr, -dY_arr));
  return true;
}

REGISTER_CPU_OPERATOR(
    Abs,
    UnaryElementwiseOp<TensorTypes<float>, CPUContext, AbsFunctor<CPUContext>>);
REGISTER_CPU_OPERATOR(
    AbsGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        AbsGradientFunctor<CPUContext>>);

OPERATOR_SCHEMA(Abs)
    .NumInputs(1)
    .NumOutputs(1)
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Calculates the absolute value of the given input tensor, element-wise.

Github Links:

- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/abs_op.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```
workspace.ResetWorkspace()

op = core.CreateOperator(
    "Abs",
    ["X"],
    ["Y"]
)

workspace.FeedBlob("X", np.random.randn(5).astype(np.float32))
print("X:", workspace.FetchBlob("X"))
workspace.RunOperatorOnce(op)
print("Y:", workspace.FetchBlob("Y"))

```

**Result**

```

X: [ 0.3005476   1.551666   -1.3591481   0.39191285 -0.21866608]
Y: [0.3005476  1.551666   1.3591481  0.39191285 0.21866608]

```

</details>

)DOC")
    .Input(0, "X", "*(type: Tensor<float>)* Input tensor.")
    .Output(
        0,
        "Y",
        "*(type: Tensor`<float>`)* Absolute value of input element-wise.")
    .InheritOnnxSchema();

OPERATOR_SCHEMA(AbsGradient)
    .NumInputs(2)
    .NumOutputs(1)
    .IdenticalTypeAndShapeOfInput(0);

namespace {

class GetAbsGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  std::vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "AbsGradient",
        "",
        std::vector<std::string>{I(0), GO(0)},
        std::vector<std::string>{GI(0)});
  }
};

} // namespace

REGISTER_GRADIENT(Abs, GetAbsGradient);

} // namespace caffe2
