#include "caffe2/operators/sqr_op.h"

#include <string>
#include <vector>

namespace caffe2 {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    Sqr,
    UnaryElementwiseOp<TensorTypes<float>, CPUContext, SqrFunctor<CPUContext>>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(Sqr)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Performs element-wise squaring ($x^2$) of input tensor.

Github Link:
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/sqr_op.cc

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

namespace {

class GetSqrGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  std::vector<OperatorDef> GetGradientDefs() override {
    Argument scale_arg;
    scale_arg.set_name("scale");
    scale_arg.set_f(2.0);
    return std::vector<OperatorDef>{CreateOperatorDef(
                                        "Scale",
                                        "",
                                        std::vector<std::string>{GO(0)},
                                        std::vector<std::string>{GO(0)},
                                        std::vector<Argument>{scale_arg}),
                                    CreateOperatorDef(
                                        "Mul",
                                        "",
                                        std::vector<std::string>{GO(0), I(0)},
                                        std::vector<std::string>{GI(0)})};
  }
};

} // namespace

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_GRADIENT(Sqr, GetSqrGradient);

} // namespace caffe2
