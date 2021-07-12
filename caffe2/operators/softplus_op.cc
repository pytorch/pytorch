#include "caffe2/operators/softplus_op.h"

#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <>
bool SoftplusOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);

  auto* Y = Output(0, X.sizes(), at::dtype<float>());

  EigenVectorMap<float>(Y->template mutable_data<float>(), X.numel()) =
      (ConstEigenVectorMap<float>(X.data<float>(), X.numel()).array().exp() +
       1.0f)
          .log();
  return true;
}

template <>
bool SoftplusGradientOp<float, CPUContext>::RunOnDevice() {
  auto& Y = Input(0);
  auto& dY = Input(1);

  DCHECK_EQ(dY.numel(), Y.numel());
  auto* dX = Output(0, Y.sizes(), at::dtype<float>());

  const float* Ydata = Y.data<float>();
  const float* dYdata = dY.data<float>();
  float* dXdata = dX->template mutable_data<float>();
  EigenVectorArrayMap<float> dXvec(dXdata, dX->numel());
  ConstEigenVectorArrayMap<float> Yvec(Ydata, Y.numel());
  ConstEigenVectorArrayMap<float> dYvec(dYdata, dY.numel());
  dXvec = dYvec * (1.0 - (-Yvec).exp());
  return true;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(Softplus, SoftplusOp<float, CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(SoftplusGradient, SoftplusGradientOp<float, CPUContext>);

// Input: X, output: Y
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(Softplus)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Softplus takes one input data tensor $X$ and produces one output data tensor $Y,$ where the softplus function, $y = ln(e^x + 1)$, is applied to $X$ elementwise.

Github Links:

- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/softplus_op.h
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/softplus_op.cc


<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "Softplus",
    ["X"],
    ["Y"],
)

workspace.FeedBlob("X", np.random.randn(3, 3).astype(np.float32))
print("X:\n", workspace.FetchBlob("X"), "\n")

workspace.RunOperatorOnce(op)
print("Y:\n", workspace.FetchBlob("Y"))

```

**Result**

```

X:
 [[-0.5380011   0.65190786  0.55673236]
 [-0.16272168  0.5451048   0.30880353]
 [-0.76606876 -0.6238556  -0.40444514]]

Y:
 [[0.4598992  1.0713093  1.0097669 ]
 [0.61509246 1.0023911  0.8594219 ]
 [0.38174385 0.42909983 0.5112337 ]]

```

</details>

)DOC")
    .Input(0, "X", "Input data blob to be operated on.")
    .Output(0, "Y", "Output data blob with same shape as input.")
    .InheritOnnxSchema();

// Input: Y, dY, output: dX
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(SoftplusGradient)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{1, 0}});

class GetSoftplusGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "SoftplusGradient",
        "",
        vector<string>{O(0), GO(0)},
        vector<string>{GI(0)});
  }
};
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_GRADIENT(Softplus, GetSoftplusGradient);

} // namespace caffe2
