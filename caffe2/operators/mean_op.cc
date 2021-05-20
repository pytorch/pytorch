#include "caffe2/operators/mean_op.h"

namespace caffe2 {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(Mean, MeanOp<CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(MeanGradient, MeanGradientOp<CPUContext>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(Mean)
    .NumInputs(1, INT_MAX)
    .NumOutputs(1)
    .IdenticalTypeAndShapeOfInput(0)
    .AllowInplace({{0, 0}})
    .SetDoc(R"DOC(
Element-wise mean of an arbitrary number of input tensors. This operation can be
performed in-place, by using the first input blob as the output blob. All inputs
must have the same shape and data type, and the output will have the same shape
as the inputs.

Github Link:
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/mean_op.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "Mean",
    ["X", "Y", "Z"],
    ["X"],
)

workspace.FeedBlob("X", (np.random.rand(3,3)).astype(np.float32))
workspace.FeedBlob("Y", (np.random.rand(3,3)).astype(np.float32))
workspace.FeedBlob("Z", (np.random.rand(3,3)).astype(np.float32))
print("X:", workspace.FetchBlob("X"))
print("Y:", workspace.FetchBlob("Y"))
print("Z:", workspace.FetchBlob("Z"))
workspace.RunOperatorOnce(op)
print("Mean:", workspace.FetchBlob("X"))

```

**Result**

```

X:
[[0.6035237  0.5305746  0.6298913 ]
 [0.9169737  0.01280353 0.16286302]
 [0.6017664  0.9946255  0.05128575]]
Y:
[[0.07544111 0.45371833 0.08460239]
 [0.9708728  0.7422064  0.7933344 ]
 [0.97671497 0.3411384  0.73818344]]
Z:
[[0.08837954 0.90187573 0.46734726]
 [0.6308827  0.8719029  0.39888734]
 [0.90059936 0.92883426 0.5695987 ]]
Mean:
[[0.25578147 0.6287229  0.39394698]
 [0.8395764  0.5423043  0.45169494]
 [0.8263602  0.75486606 0.45302266]]

```

</details>

)DOC")
    .Input(0, "X, Y, ...", "*(type: Tensor`<Ord>`)* List of input tensors with the same shape.")
    .Output(0, "M", "*(type: Tensor`<Ord>`)* Output tensor with the same dimensions as inputs. Contains "
    "the mean values of the input tensors calculated element-wise.");

class GetMeanGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    auto outputs = std::vector<string>();
    for (int i = 0; i < def_.input_size(); i++) {
      outputs.push_back(GI(i));
    }
    return SingleGradientDef(
        "MeanGradient", "", std::vector<string>{GO(0)}, outputs);
  }
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_GRADIENT(Mean, GetMeanGradient);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(MeanGradient)
    .NumInputs(1)
    .NumOutputs(1, INT_MAX)
    .AllowInplace({{0, 0}});

} // namespace caffe2
