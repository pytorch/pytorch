#include "caffe2/operators/sqrt_op.h"

#include <string>
#include <vector>

namespace caffe2 {

REGISTER_CPU_OPERATOR(
    Sqrt,
    UnaryElementwiseOp<
        TensorTypes<float, double>,
        CPUContext,
        SqrtFunctor<CPUContext>>);

// Input: X, output: Y
OPERATOR_SCHEMA(Sqrt)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Performs element-wise square-root ($\sqrt{x}$) of input tensor $X$.

Github Link:
- https://github.com/pytorch/pytorch/blob/main/caffe2/operators/sqrt_op.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "Sqrt",
    ["X"],
    ["Y"],
)

workspace.FeedBlob("X", (np.random.randint(10, size=(3,3))).astype(np.float32))
print("X:", workspace.FetchBlob("X"))
workspace.RunOperatorOnce(op)
print("Y:", workspace.FetchBlob("Y"))

```

**Result**

```

X:
[[8. 3. 3.]
 [4. 0. 0.]
 [1. 2. 5.]]
Y:
[[2.8284268  1.7320508  1.7320508 ]
 [1.9999999  0.         0.        ]
 [0.99999994 1.4142134  2.236068  ]]

```

</details>
)DOC")
.Input(0, "X", "*(type: Tensor`<float>`)* Input data tensor.")
.Output(0, "Y", "*(type: Tensor`<float>`)* Output tensor.");

namespace {

class GetSqrtGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  std::vector<OperatorDef> GetGradientDefs() override {
    Argument scale_arg;
    scale_arg.set_name("scale");
    scale_arg.set_f(0.5);
    return std::vector<OperatorDef>{CreateOperatorDef(
                                        "Scale",
                                        "",
                                        std::vector<std::string>{GO(0)},
                                        std::vector<std::string>{GI(0)},
                                        std::vector<Argument>{scale_arg}),
                                    CreateOperatorDef(
                                        "Div",
                                        "",
                                        std::vector<std::string>{GI(0), O(0)},
                                        std::vector<std::string>{GI(0)})};
  }
};

} // namespace

REGISTER_GRADIENT(Sqrt, GetSqrtGradient);

} // namespace caffe2
