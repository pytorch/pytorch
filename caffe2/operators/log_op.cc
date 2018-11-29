#include "caffe2/operators/log_op.h"

#include <string>
#include <vector>

namespace caffe2 {

REGISTER_CPU_OPERATOR(
    Log,
    UnaryElementwiseOp<TensorTypes<float>, CPUContext, LogFunctor<CPUContext>>);

namespace {
// Since the actual flops of the non-linear operator depends on the
// implementation, we use the number of non-linear operations as the proxy for
// the analytical flops for non-linear operator
OpSchema::Cost CostInferenceForLog(
    const OperatorDef& def,
    const vector<TensorShape>& in) {
  CAFFE_ENFORCE_EQ(in.size(), 1, "Log requires one input");
  struct OpSchema::Cost c;
  ArgumentHelper helper(def);

  const uint64_t input_size =
      size_to_dim_(in[0].dims().size(), GetDimsVector(in[0]));

  const auto& X = in[0];
  c.flops = input_size;
  c.bytes_read = input_size * sizeof(X.data_type());
  c.bytes_written = input_size * sizeof(X.data_type());
  c.params_bytes = 0;
  return c;
}
} // namespace

using namespace std::placeholders;
OPERATOR_SCHEMA(Log)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .CostInferenceFunction(std::bind(CostInferenceForLog, _1, _2))
    .SetDoc(R"DOC(
Calculates the natural log of the given input tensor ($ln(x)$), element-wise. This
operation can be done in an in-place fashion too, by providing the same input
and output blobs.

Github Link:
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/log_op.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "Log",
    ["X"],
    ["X"],
)

workspace.FeedBlob("X", (np.random.rand(3,3)).astype(np.float32))
print("X before running op:", workspace.FetchBlob("X"))
workspace.RunOperatorOnce(op)
print("X after running op:", workspace.FetchBlob("X"))

```

**Result**

```

X before running op:
[[0.07341351 0.15404125 0.386613  ]
 [0.34090295 0.99727786 0.24141751]
 [0.32016268 0.8724168  0.93515724]]
X after running op:
[[-2.6116474  -1.8705349  -0.9503311 ]
 [-1.0761575  -0.00272586 -1.4212275 ]
 [-1.138926   -0.13648799 -0.06704059]]

```

</details>

)DOC")
    .Input(0, "X", "*(type: Tensor`<float>`)* Input tensor.")
    .Output(
        0,
        "Y",
        "*(type: Tensor`<float>`)* Output tensor computed as the natural log of the input tensor computed, element-wise.")
    .InheritOnnxSchema();

namespace {

class GetLogGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  std::vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "Div",
        "",
        std::vector<std::string>{GO(0), I(0)},
        std::vector<std::string>{GI(0)});
  }
};

} // namespace

REGISTER_GRADIENT(Log, GetLogGradient);

} // namespace caffe2
