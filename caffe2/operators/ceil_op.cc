#include "caffe2/operators/ceil_op.h"

#include "caffe2/utils/math.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(Ceil, CeilOp<float, CPUContext>);

OPERATOR_SCHEMA(Ceil)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .SetDoc(R"DOC(
Element-wise application of the ceil function ($y=ceil(x)$) to the input tensor
`X`. Output tensor shape is the same as the input tensor.

Github Link:
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/ceil_op.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "Ceil",
    ["X"],
    ["X"],
)

workspace.FeedBlob("X", (np.random.uniform(-10, 10, (5,5))).astype(np.float32))
print("X before running op:", workspace.FetchBlob("X"))
workspace.RunOperatorOnce(op)
print("X after running op:", workspace.FetchBlob("X"))

```

**Result**

```

X before running op:
[[ 8.44598    -6.5098248  -2.2993476  -7.6859694   0.58566964]
 [-7.846551   -0.03689406  6.9362907  -4.0521703   4.4969673 ]
 [ 0.33355865 -7.895527   -8.393201    9.374202   -2.3930092 ]
 [-6.3061996   3.1403487   3.782099   -8.516556   -2.8387244 ]
 [-2.0164998   4.7663913  -3.422966    0.3636999   8.75713   ]]
X after running op:
[[ 9. -6. -2. -7.  1.]
 [-7. -0.  7. -4.  5.]
 [ 1. -7. -8. 10. -2.]
 [-6.  4.  4. -8. -2.]
 [-2.  5. -3.  1.  9.]]

```

</details>

)DOC")
    .Input(0, "X", "*(type: Tensor`<float>`)* Input tensor.")
    .Output(0, "Y", "*(type: Tensor`<float>`)* Output tensor.");

// TODO: Write gradient for this when needed
GRADIENT_NOT_IMPLEMENTED_YET(Ceil);

} // namespace caffe2
