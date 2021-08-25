#include "is_empty_op.h"
namespace caffe2 {

REGISTER_CPU_OPERATOR(IsEmpty, IsEmptyOp<CPUContext>);

OPERATOR_SCHEMA(IsEmpty)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
The *IsEmpty* op accepts a single input $tensor$, and produces a single boolean output $is\_empty$. The output is *True* if and only if $tensor$ has size == 0.

Github Links:

- https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc
- https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.h


<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "IsEmpty",
    ["tensor"],
    ["is_empty"],
)

// Use a not-empty tensor
workspace.FeedBlob("tensor", np.random.randn(2, 2).astype(np.float32))
print("tensor:\n", workspace.FetchBlob("tensor"))

workspace.RunOperatorOnce(op)
print("is_empty: ", workspace.FetchBlob("is_empty"),"\n")

// Use an empty tensor
workspace.FeedBlob("tensor", np.empty(0))
print("tensor:\n", workspace.FetchBlob("tensor"))

workspace.RunOperatorOnce(op)
print("is_empty: ", workspace.FetchBlob("is_empty"))

```

**Result**

```

tensor:
 [[ 0.26018378  0.6778789 ]
 [-1.3097627  -0.40083608]]
is_empty:  False

tensor:
 []
is_empty:  True

```

</details>

)DOC")
    .ScalarType(::caffe2::TensorProto_DataType::TensorProto_DataType_BOOL)
    .Input(0, "tensor", "Input data tensor to check if empty.")
    .Output(
        0,
        "is_empty",
        "Output scalar boolean tensor. True if input has size == 0.");

} // namespace caffe2
