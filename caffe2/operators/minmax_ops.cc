#include "caffe2/operators/minmax_ops.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(Min, MinOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(Max, MaxOp<float, CPUContext>);

OPERATOR_SCHEMA(Max)
    .NumInputs(1, INT_MAX)
    .NumOutputs(1)
    .IdenticalTypeAndShapeOfInput(0)
    .AllowInplace({{0, 0}})
    .SetDoc(R"DOC(
Element-wise max of an arbitrary number of input tensors. This operation can be
performed in-place, by using the first input blob as the output blob. All inputs
must have the same shape and data type, and the output will have the same shape
as the inputs.

Github Link:
- https://github.com/pytorch/pytorch/blob/main/caffe2/operators/minmax_ops.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "Max",
    ["X", "Y", "Z"],
    ["X"],
)

workspace.FeedBlob("X", (np.random.rand(3,3)).astype(np.float32))
workspace.FeedBlob("Y", (np.random.rand(3,3)).astype(np.float32))
workspace.FeedBlob("Z", (np.random.rand(3,3)).astype(np.float32))
print("X:", workspace.FetchBlob("X"))
print("Y:", workspace.FetchBlob("Y"))
print("Z:", workspace.FetchBlob("Z"))
workspace.RunOperatorOnce(op)
print("Max:", workspace.FetchBlob("X"))

```

**Result**

```

X:
[[0.4496477  0.07061381 0.7139333 ]
 [0.83203    0.05970785 0.72786295]
 [0.75988126 0.04601283 0.32820013]]
Y:
[[0.05683139 0.16872478 0.671098  ]
 [0.70739156 0.09878621 0.03416285]
 [0.34087983 0.94986707 0.67263436]]
Z:
[[0.48051122 0.07141234 0.85264146]
 [0.77086854 0.22082241 0.13154659]
 [0.42401117 0.995431   0.4263775 ]]
Max:
[[0.48051122 0.16872478 0.85264146]
 [0.83203    0.22082241 0.72786295]
 [0.75988126 0.995431   0.67263436]]

```

</details>

)DOC")
    .Input(
        0,
        "X, Y, ...",
        "*(type: Tensor`<Ord>`)* List of input tensors with the same shape.")
    .Output(
        0,
        "M",
        "*(type: Tensor`<Ord>`)* Output tensor with same dimensions as input(s)."
        "Contains the maximum valued element at each location.")
    .InheritOnnxSchema();

OPERATOR_SCHEMA(Min)
    .NumInputs(1, INT_MAX)
    .NumOutputs(1)
    .IdenticalTypeAndShapeOfInput(0)
    .AllowInplace({{0, 0}})
    .SetDoc(R"DOC(
Element-wise min of an arbitrary number of input tensors. This operation can be performed in-place, by using the first input blob as the output blob. All inputs must have the same shape and data type, and the output will have the same shape as the inputs.

Github Link:
- https://github.com/pytorch/pytorch/blob/main/caffe2/operators/minmax_ops.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "Min",
    ["X", "Y", "Z"],
    ["X"],
)

workspace.FeedBlob("X", (np.random.rand(2,2)).astype(np.float32))
workspace.FeedBlob("Y", (np.random.rand(2,2)).astype(np.float32))
workspace.FeedBlob("Z", (np.random.rand(2,2)).astype(np.float32))
print("X:", workspace.FetchBlob("X"))
print("Y:", workspace.FetchBlob("Y"))
print("Z:", workspace.FetchBlob("Z"))
workspace.RunOperatorOnce(op)
print("Min:", workspace.FetchBlob("X"))

```

**Result**

```

X:
[[0.32731926 0.4939747 ]
 [0.29242373 0.43460014]]
Y:
[[0.40928316 0.916115  ]
 [0.77526504 0.29339448]]
Z:
[[0.7899794  0.90335774]
 [0.82599413 0.2843068 ]]
Min:
[[0.32731926 0.4939747 ]
 [0.29242373 0.2843068 ]]

```

</details>

)DOC")
    .Input(
        0,
        "X, Y, ...",
        "*(type: Tensor`<Ord>`)* List of input tensors with the same shape.")
    .Output(
        0,
        "M",
        "*(type: Tensor`<Ord>`)* Output tensor with same dimensions as input(s)."
        "Contains the minimum valued element at each location.")
    .InheritOnnxSchema();

} // namespace caffe2
