#include "caffe2/operators/assert_op.h"

namespace caffe2 {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(Assert, AssertOp<CPUContext>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(Assert)
    .NumInputs(1)
    .NumOutputs(0)
    .SetDoc(R"DOC(
Takes in a tensor of type *bool*, *int*, *long*, or *long long* and checks if all values are True when coerced into a boolean. In other words, for non-bool types this asserts that all values in the tensor are non-zero. If a value is False after coerced into a boolean, the operator throws an error. Else, if all values are True, nothing is returned. For tracability, a custom error message can be set using the `error_msg` argument.

Github Links:
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/assert_op.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "Assert",
    ["A"],
    [],
    error_msg="Failed assertion from Assert operator"
)

workspace.FeedBlob("A", np.random.randint(10, size=(3,3)).astype(np.int32))
print("A:", workspace.FetchBlob("A"))
try:
    workspace.RunOperatorOnce(op)
except RuntimeError:
    print("Assertion Failed!")
else:
    print("Assertion Passed!")

```

**Result**

```

A:
[[7 5 6]
 [1 2 4]
 [5 3 7]]
Assertion Passed!

```

</details>

        )DOC")
    .Arg(
        "error_msg",
        "(*string*): custom error message to be thrown when the input does not pass assertion",
        false)
    .Input(0,"X","(*Tensor*): input tensor");

} // namespace caffe2
