#include "caffe2/operators/clip_op.h"
#include "caffe2/utils/eigen_utils.h"

namespace caffe2 {

template <>
bool ClipOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);

  auto* Y = Output(0, X.sizes(), at::dtype<float>());
  EigenVectorMap<float>(Y->template mutable_data<float>(), Y->numel()) =
      ConstEigenVectorMap<float>(X.data<float>(), X.numel())
          .cwiseMax(min_)
          .cwiseMin(max_);
  return true;
}

template <>
bool ClipGradientOp<float, CPUContext>::RunOnDevice() {
  auto& Y = Input(0);
  auto& dY = Input(1);

  CAFFE_ENFORCE_GE(Y.numel(), 0);
  CAFFE_ENFORCE_EQ(dY.numel(), Y.numel());
  auto* dX = Output(0, Y.sizes(), at::dtype<float>());
  const float* Ydata = Y.data<float>();
  const float* dYdata = dY.data<float>();
  float* dXdata = dX->template mutable_data<float>();
  for (int i = 0; i < Y.numel(); ++i) {
    // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
    dXdata[i] = dYdata[i] * (Ydata[i] > min_ && Ydata[i] < max_);
  }
  return true;
}

REGISTER_CPU_OPERATOR(Clip, ClipOp<float, CPUContext>);
REGISTER_CPU_GRADIENT_OPERATOR(ClipGradient, ClipGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(Clip)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
This operator limits the given input within an interval. The interval is
specified by the `min` and `max` arguments. They default to
*numeric_limits::lowest()* and *numeric_limits::max()* respectively. The
clipping operation can be done in an in-place fashion by using the same output
blob as the input blob.

Github Links:

- https://github.com/pytorch/pytorch/blob/main/caffe2/operators/clip_op.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```
workspace.ResetWorkspace()

op = core.CreateOperator(
    "Clip",
    ["X"],
    ["Y"],
    min=20.0,
    max=60.0

)

workspace.FeedBlob("X", (np.random.randint(100, size=(5,5))).astype(np.float32))
print("X:", workspace.FetchBlob("X"))
workspace.RunOperatorOnce(op)
print("Y:", workspace.FetchBlob("Y"))

```

**Result**

```
X: [[45. 16. 59. 99. 48.]
 [12. 44. 46. 82. 28.]
 [ 1. 91. 18.  9. 71.]
 [24. 37. 61. 12. 81.]
 [36. 38. 30. 84. 40.]]
Y: [[45. 20. 59. 60. 48.]
 [20. 44. 46. 60. 28.]
 [20. 60. 20. 20. 60.]
 [24. 37. 60. 20. 60.]
 [36. 38. 30. 60. 40.]]
```

</details>

)DOC")
    .Arg(
        "min",
        "*(type: float)* Minimum value, under which element is "
        "replaced by min (default=*numeric_limits::lowest()*).")
    .Arg(
        "max",
        "*(type: float)* Maximum value, under which element is "
        "replaced by max (default=*numeric_limits::max()*).")
    .Input(
        0,
        "X",
        "*(Tensor`<float>`)* Input tensor within range "
        "[*numeric_limits::lowest()*, *numeric_limits::max()*].")
    .Output(
        0,
        "Y",
        "*(Tensor`<float>`)* Output tensor clipped within range [`min`, `max`].")
    .InheritOnnxSchema();

GRADIENT_OPERATOR_SCHEMA(ClipGradient)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{1, 0}});

class GetClipGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "ClipGradient", "",
        vector<string>{O(0), GO(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(Clip, GetClipGradient);
}  // namespace caffe2
