#include "caffe2/operators/find_duplicate_elements_op.h"

namespace caffe2 {
namespace {
REGISTER_CPU_OPERATOR(
    FindDuplicateElements,
    FindDuplicateElementsOp<CPUContext>);

OPERATOR_SCHEMA(FindDuplicateElements)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
The *FindDuplicateElements* op takes a single 1-D tensor *data* as input and returns a single 1-D output tensor *indices*. The output tensor contains the indices of the duplicate elements of the input, excluding the first occurrences. If all elements of *data* are unique, *indices* will be empty.

Github Links:

- https://github.com/caffe2/caffe2/blob/master/caffe2/operators/find_duplicate_elements_op.h
- https://github.com/caffe2/caffe2/blob/master/caffe2/operators/find_duplicate_elements_op.cc


<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "FindDuplicateElements",
    ["data"],
    ["indices"],
)

workspace.FeedBlob("data", np.array([8,2,1,1,7,8,1]).astype(np.float32))
print("data:\n", workspace.FetchBlob("data"))

workspace.RunOperatorOnce(op)
print("indices: \n", workspace.FetchBlob("indices"))

```

**Result**

```

data:
 [8. 2. 1. 1. 7. 8. 1.]
indices:
 [3 5 6]

```

</details>


  )DOC")
    .Input(0, "data", "a 1-D tensor.")
    .Output(
        0,
        "indices",
        "Indices of duplicate elements in data, excluding first occurrences.");

SHOULD_NOT_DO_GRADIENT(FindDuplicateElements);
} // namespace
} // namespace caffe2
