#include "caffe2/operators/math_ops.h"
#include "caffe2/utils/math.h"


namespace caffe2 {

struct LogCPUFunctor {
  template <typename T>
  inline void
  operator()(const int n, const T* x, T* y, CPUContext* device_context) {
    math::Log<T, CPUContext>(n, x, y, device_context);
  }
};

REGISTER_CPU_OPERATOR(
    Log,
    UnaryElementwiseOp<TensorTypes<float>, CPUContext, LogCPUFunctor>);

OPERATOR_SCHEMA(Log)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
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
    .InheritOnnxSchema("Log");

class GetLogGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "Div",
        "",
        std::vector<string>{GO(0), I(0)},
        std::vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(Log, GetLogGradient);

} // namespace caffe2
