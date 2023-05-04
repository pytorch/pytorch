#include "caffe2/operators/utility_ops.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(Sum, SumOp<CPUContext>);

OPERATOR_SCHEMA(Sum)
    .NumInputs(1, INT_MAX)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .CostInferenceFunction(CostInferenceForSum)
    .InputsCanCrossDevices()
    .IdenticalTypeAndShapeOfInput(0)
    .SetDoc(R"DOC(
Element-wise sum of each of the input tensors. The first input tensor can be used
in-place as the output tensor, in which case the sum will be done in place and
results will be accumulated the first input tensor. All inputs and outputs must
have the same shape and data type.

Github Links:

- https://github.com/pytorch/pytorch/blob/main/caffe2/operators/elementwise_sum_op.cc


<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "Sum",
    ["A",  "B"],
    ["C"],
)

workspace.FeedBlob("A", np.array([[1,2],[3,4]]).astype(np.float32))
workspace.FeedBlob("B", np.array([[5,6],[7,8]]).astype(np.float32))
print("A:", workspace.FetchBlob("A"))
print("B:", workspace.FetchBlob("B"))
workspace.RunOperatorOnce(op)
print("C:", workspace.FetchBlob("A"))

```

**Result**

```

A: [[1. 2.]
 [3. 4.]]
B: [[5. 6.]
 [7. 8.]]
C: [[1. 2.]
 [3. 4.]]

```

</details>

<details>

<summary> <b>Example 2</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "Sum",
    ["A",  "B"],
    ["A"],  // inplace
)

workspace.FeedBlob("A", np.array([[1,2,5],[8,3,4]]).astype(np.float32))
workspace.FeedBlob("B", np.array([[9,5,6],[6,7,8]]).astype(np.float32))
print("A:", workspace.FetchBlob("A"))
print("B:", workspace.FetchBlob("B"))
workspace.RunOperatorOnce(op)
print("A after Sum:", workspace.FetchBlob("A"))

```

**Result**

```

A: [[1. 2. 5.]
 [8. 3. 4.]]
B: [[9. 5. 6.]
 [6. 7. 8.]]
A after Sum: [[10.  7. 11.]
 [14. 10. 12.]]

```

</details>

)DOC")
    .Input(
        0,
        "A",
        "*(type: Tensor`<float>`)* First tensor to be added element-wise.")
    .Input(
        1,
        "B",
        "*(type: Tensor`<float>`)* Second tensor to be added element-wise.")
    .Output(0, "C", "*(type: Tensor`<float>`)* Sum of A and B.")
    .InheritOnnxSchema();
}
