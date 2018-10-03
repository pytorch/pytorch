#include "caffe2/operators/add5_op.h"

#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {

template <>
template <typename T>
bool Add5Op<CPUContext>::DoRunWithType() {
  const auto& data = Input(DATA);
  auto N = data.size();
  const auto* data_ptr = data.template data<T>();
  auto* output = Output(0);
  output->ResizeLike(data);
  auto* output_ptr = output->template mutable_data<T>();

  for (auto i = 0; i < N; i++) {
    // TODO - 1
  }
  return true;
}

template <>
template <typename T>
bool Add5GradientOp<CPUContext>::DoRunWithType() {
  const auto& data = Input(DATA);
  auto N = data.size();
  const auto* data_ptr = data.template data<T>();
  auto* output = Output(0);
  output->ResizeLike(data);
  auto* output_ptr = output->template mutable_data<T>();

  for (auto i = 0; i < N; i++) {
    // GI[0] = GO[0]
    // TODO - 2
  }
  return true;
}

namespace {

REGISTER_CPU_OPERATOR(Add5, Add5Op<CPUContext>);
OPERATOR_SCHEMA(Add5)
    .NumInputs(1)
    .NumOutputs(1)
    .IdenticalTypeAndShape()
    .AllowInplace({{0, 0}})
    .SetDoc(R"DOC(
Element-wise add 5 operation. Each element in the output equals to the
corresponding element in the input data.

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "Add5",
    ["X"],
    ["Y"],
)

workspace.FeedBlob("X", (np.random.randint(100, size=(5,5))))
print("X before running op:", workspace.FetchBlob("X"))
workspace.RunOperatorOnce(op)
print("X after running op:", workspace.FetchBlob("Y"))

```

**Result**

```

X before running op:

X after running op:
[[6 2 3 3 0]
 [4 5 8 0 5]
 [4 6 4 3 6]
 [0 6 7 2 8]
 [1 4 6 7 5]]

```

 </details>

)DOC")
    .Input(0, "X", "Input tensor.")
    .Output(0, "Y", "Output tensor");


REGISTER_CPU_OPERATOR(Add5Gradient, Add5GradientOp<CPUContext>);
OPERATOR_SCHEMA(Add5Gradient)
    .NumInputs(1)
    .NumOutputs(1);

class GetAdd5Gradient final : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;

  std::vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "Add5Gradient",
        "",
        std::vector<std::string>{GO(0)},
        std::vector<std::string>{GI(0)});
  }
};

REGISTER_GRADIENT(Add5, GetAdd5Gradient);


} // namespace
} // namespace caffe2
