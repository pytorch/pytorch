#include "caffe2/operators/mod_op.h"

#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {

template <>
template <typename T>
bool ModOp<CPUContext>::DoRunWithType() {
  auto& data = Input(DATA);
  auto N = data.numel();
  const auto* data_ptr = data.template data<T>();

  auto* output = Output(0, Input(DATA).sizes(), at::dtype<T>());
  auto* output_ptr = output->template mutable_data<T>();

  for (auto i = 0; i < N; i++) {
    output_ptr[i] = data_ptr[i] % divisor_;
    if (output_ptr[i] && sign_follow_divisor_ &&
        ((output_ptr[i] > 0) != (divisor_ > 0))) {
      output_ptr[i] += divisor_;
    }
  }
  return true;
}

REGISTER_CPU_OPERATOR(Mod, ModOp<CPUContext>);
OPERATOR_SCHEMA(Mod)
    .NumInputs(1)
    .NumOutputs(1)
    .Arg("divisor", "*(type: int; default: 0)* Divisor of the modulo operation (must be >= 1).")
    .Arg(
        "sign_follow_divisor",
        "*(type: bool; default: False)* If true, sign of output matches divisor, else if false, sign follows dividend.")
    .IdenticalTypeAndShape()
    .AllowInplace({{0, 0}})
    .SetDoc(R"DOC(
Element-wise modulo operation. Each element in the output is the modulo result
of the corresponding element in the input data. The divisor of the modulo is
provided by the `divisor` argument.

Github Link:
- https://github.com/pytorch/pytorch/blob/main/caffe2/operators/mod_op.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "Mod",
    ["X"],
    ["Y"],
    divisor=10
)

workspace.FeedBlob("X", (np.random.randint(100, size=(5,5))))
print("X before running op:", workspace.FetchBlob("X"))
workspace.RunOperatorOnce(op)
print("X after running op:", workspace.FetchBlob("Y"))

```

**Result**

```

X before running op:
[[56 22 43 13 60]
 [ 4 55 58 10 45]
 [64 66  4  3 66]
 [10 36 47 52 78]
 [91  4 36 47 95]]
X after running op:
[[6 2 3 3 0]
 [4 5 8 0 5]
 [4 6 4 3 6]
 [0 6 7 2 8]
 [1 4 6 7 5]]

 ```

 </details>

)DOC")
    .Input(0, "X", "*(type: Tensor`<int>`)* Input tensor with int32 or int64 data.")
    .Output(0, "Y", "*(type: Tensor`<int>`)* Output tensor of data with modulo operation applied.");

SHOULD_NOT_DO_GRADIENT(ModOp);
} // namespace caffe2
