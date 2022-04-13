#include "caffe2/operators/exp_op.h"

#include <string>
#include <vector>

namespace caffe2 {

REGISTER_CPU_OPERATOR(
    Exp,
    UnaryElementwiseOp<TensorTypes<float>, CPUContext, ExpFunctor<CPUContext>>);

OPERATOR_SCHEMA(Exp)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Calculates the exponential of the given input tensor ($exp(x)$), element-wise. This
operation can be done in an in-place fashion too, by providing the same input
and output blobs.

Github Link:
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/exp_op.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "Exp",
    ["X"],
    ["X"],
)

workspace.FeedBlob("X", (np.random.rand(3,3)).astype(np.float32))
print("X before running op:", workspace.FetchBlob("X"))
workspace.RunOperatorOnce(op)
print("X after running op:", workspace.FetchBlob("X"))

```

**Result**

```

X before running op:
[[0.5821691  0.07719802 0.50159824]
 [0.40952456 0.36788362 0.84887683]
 [0.02472685 0.65730894 0.9066397 ]]
X after running op:
[[1.7899168 1.080256  1.6513585]
 [1.5061016 1.4446739 2.3370204]
 [1.0250351 1.9295927 2.4759884]]

```

</details>

)DOC")
    .Input(0, "X", "*(type: Tensor`<float>`)* Input tensor.")
    .Output(
        0,
        "Y",
        "*(type: Tensor`<float>`)* The exponential of the input tensor computed "
        "element-wise.")
    .InheritOnnxSchema();

namespace {

class GetExpGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  std::vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "Mul",
        "",
        std::vector<std::string>{O(0), GO(0)},
        std::vector<std::string>{GI(0)});
  }
};

} // namespace

REGISTER_GRADIENT(Exp, GetExpGradient);

} // namespace caffe2
