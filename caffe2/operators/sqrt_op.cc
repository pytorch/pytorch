#include <Eigen/Core>
#include "caffe2/operators/elementwise_op.h"

namespace caffe2 {

struct SqrtCPUFunctor {
  template <typename T>
  inline void
  operator()(const int n, const T* x, T* y, CPUContext* /*device_context*/) {
    EigenVectorArrayMap<T>(y, n) = ConstEigenVectorArrayMap<T>(x, n).sqrt();
  }
};

REGISTER_CPU_OPERATOR(
    Sqrt,
    UnaryElementwiseOp<TensorTypes<float>, CPUContext, SqrtCPUFunctor>);
// Input: X, output: Y
OPERATOR_SCHEMA(Sqrt)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Performs element-wise square-root ($\sqrt{x}$) of input tensor $X$.

Github Link:
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/sqrt_op.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "Sqrt",
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
[[8. 3. 3.]
 [4. 0. 0.]
 [1. 2. 5.]]
Y:
[[2.8284268  1.7320508  1.7320508 ]
 [1.9999999  0.         0.        ]
 [0.99999994 1.4142134  2.236068  ]]

```

</details>
)DOC")
.Input(0, "X", "*(type: Tensor`<float>`)* Input data tensor.")
.Output(0, "Y", "*(type: Tensor`<float>`)* Output tensor.");

class GetSqrtGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    Argument scale_arg;
    scale_arg.set_name("scale");
    scale_arg.set_f(0.5);
    return vector<OperatorDef>{CreateOperatorDef(
                                   "Scale",
                                   "",
                                   std::vector<string>{GO(0)},
                                   std::vector<string>{GI(0)},
                                   std::vector<Argument>{scale_arg}),
                               CreateOperatorDef(
                                   "Div",
                                   "",
                                   std::vector<string>{GI(0), O(0)},
                                   std::vector<string>{GI(0)})};
  }
};
REGISTER_GRADIENT(Sqrt, GetSqrtGradient);
} // namespace caffe2
