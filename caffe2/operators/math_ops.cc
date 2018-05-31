#include "caffe2/operators/math_ops.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

struct SqrCPUFunctor {
  template <typename T>
  inline void
  operator()(const int n, const T* x, T* y, CPUContext* device_context) {
    math::Sqr<T, CPUContext>(n, x, y, device_context);
  }
};

REGISTER_CPU_OPERATOR(
    Sqr,
    UnaryElementwiseOp<TensorTypes<float>, CPUContext, SqrCPUFunctor>);

OPERATOR_SCHEMA(Sqr)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Performs element-wise squaring ($x^2$) of input tensor.

Github Link:
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/math_ops.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "Sqr",
    ["X"],
    ["Y"],
)

workspace.FeedBlob("X", (np.random.randint(10, size=(3,3))).astype(np.float32))
print("X:", workspace.FetchBlob("X"))
workspace.RunOperatorOnce(op)
print("Y:", workspace.FetchBlob("Y"))

```

**Result**

```

X:
[[4. 6. 2.]
 [0. 1. 6.]
 [9. 2. 7.]]
Y:
[[16. 36.  4.]
 [ 0.  1. 36.]
 [81.  4. 49.]]

```

</details>

    )DOC")
    .Input(0, "X", "*(type: Tensor`<float>`)* Input data tensor.")
    .Output(0, "Y", "*(type: Tensor`<float>`)* Output tensor.");

class GetSqrGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    Argument scale_arg;
    scale_arg.set_name("scale");
    scale_arg.set_f(2.0);
    return vector<OperatorDef>{CreateOperatorDef(
                                   "Scale",
                                   "",
                                   std::vector<string>{GO(0)},
                                   std::vector<string>{GO(0)},
                                   std::vector<Argument>{scale_arg}),
                               CreateOperatorDef(
                                   "Mul",
                                   "",
                                   std::vector<string>{GO(0), I(0)},
                                   std::vector<string>{GI(0)})};
  }
};
REGISTER_GRADIENT(Sqr, GetSqrGradient);

struct SignCPUFunctor {
  template <typename T>
  inline void
  operator()(const int n, const T* x, T* y, CPUContext* device_context) {
    for (int i = 0; i < n; ++i) {
      y[i] = (-T(1) * (x[i] < 0)) + (x[i] > 0);
    }
  }
};

REGISTER_CPU_OPERATOR(
    Sign,
    UnaryElementwiseOp<TensorTypes<float>, CPUContext, SignCPUFunctor>);

OPERATOR_SCHEMA(Sign)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Computes sign for each element of the input: -1, 0 or 1.

Github Link:
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/math_ops.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "Sign",
    ["X"],
    ["Y"],
)

workspace.FeedBlob("X", (np.random.rand(3, 3).astype(np.float32) - np.random.rand(3, 3).astype(np.float32)))
print("X:", workspace.FetchBlob("X"))
workspace.RunOperatorOnce(op)
print("Y:", workspace.FetchBlob("Y"))

```

**Result**

```

X:
[[ 0.02816287  0.22408086 -0.30342305]
 [-0.18481976  0.03948995  0.39698976]
 [-0.63304734 -0.6919183  -0.31524038]]
Y:
[[ 1.  1. -1.]
 [-1.  1.  1.]
 [-1. -1. -1.]]

```

</details>

    )DOC")
    .Input(0, "X", "*(type: Tensor`<float>`)* Input data tensor.")
    .Output(0, "Y", "*(type: Tensor`<float>`)* Output tensor.")
    .IdenticalTypeAndShape();
SHOULD_NOT_DO_GRADIENT(Sign);

} // namespace caffe2
