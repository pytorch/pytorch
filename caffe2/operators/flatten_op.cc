#include "caffe2/operators/flatten_op.h"

namespace caffe2 {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(Flatten, FlattenOp<CPUContext>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(Flatten)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction(TensorInferenceForFlatten)
    .SetDoc(R"DOC(
Flattens the input tensor into a 2D matrix. If input tensor has shape
$(d_0, d_1, ..., d_n)$ then the output will have shape
$\bigl((d_0 * d_1 * ... * d_{(axis-1)}), (d_{axis} * d_{(axis+1)} * ... * d_n)\bigr)$.

Github Links:

- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/flatten_op.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```
workspace.ResetWorkspace()

op = core.CreateOperator(
    "Flatten",
    ["X"],
    ["Y"],
    axis=1
)

workspace.FeedBlob("X", np.random.rand(1,3,2,2))
print("X:", workspace.FetchBlob("X"))
workspace.RunOperatorOnce(op)
print("Y:", workspace.FetchBlob("Y"))
```

**Result**

```
X: [[[[0.53432311 0.23734561]
   [0.56481598 0.52152617]]

  [[0.33662627 0.32472711]
   [0.17939016 0.97175851]]

  [[0.87226421 0.49045439]
   [0.92470531 0.30935077]]]]
Y: [[0.53432311 0.23734561 0.56481598 0.52152617 0.33662627 0.32472711
  0.17939016 0.97175851 0.87226421 0.49045439 0.92470531 0.30935077]]
```

</details>

)DOC")
    .Input(0, "X", "*(type: Tensor)* Input Tensor of rank >= axis.")
    .Output(
        0,
        "Y",
        "*(type: Tensor)* A 2D tensor with the contents of the input tensor, "
        "with input dimensions up to `axis` flattened to the outer dimension "
        "of the output and the remaining input dimensions flattened into the "
        "inner dimension of the output.")
    .Arg(
        "axis",
        "*(type: int; default: 1)* Indicates up to which input dimensions "
        "(exclusive) should be flattened to the outer dimension of the output.")
    .InheritOnnxSchema();

class GetFlattenGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "ResizeLike", "", vector<string>{GO(0), I(0)}, vector<string>{GI(0)});
  }
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_GRADIENT(Flatten, GetFlattenGradient);

} // namespace caffe2
