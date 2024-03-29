#include "caffe2/operators/reciprocal_op.h"

#include <string>
#include <vector>

namespace caffe2 {

REGISTER_CPU_OPERATOR(
    Reciprocal,
    UnaryElementwiseOp<
        TensorTypes<float, double>,
        CPUContext,
        ReciprocalFunctor<CPUContext>>);

// Input: X, output: Y
OPERATOR_SCHEMA(Reciprocal)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Performs element-wise reciprocal ($\1/x$) of input tensor $X$.

Github Link:
- https://github.com/pytorch/pytorch/blob/main/caffe2/operators/reciprocal_op.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "Reciprocal",
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
[[0.125 0.3333333  0.3333333 ]
 [0.25  inf        inf       ]
 [1     0.5        0.2       ]]

```

</details>
)DOC")
.Input(0, "X", "*(type: Tensor`<float, double>`)* Input data tensor.")
.Output(0, "Y", "*(type: Tensor`<float, double>`)* Output tensor.");

OPERATOR_SCHEMA(ReciprocalGradient).NumInputs(2).NumOutputs(1).AllowInplace({{1, 0}});

} // namespace caffe2
