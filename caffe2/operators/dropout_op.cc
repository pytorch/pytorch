#include "caffe2/operators/dropout_op.h"

namespace caffe2 {

template <>
bool DropoutOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0, X.sizes(), at::dtype<float>());

  if (is_test_) {
    if (!IsInputOutputAlias(0, 0)) {
      context_.CopyFromCPU<float>(
          X.numel(), X.data<float>(), Y->template mutable_data<float>());
    }
    return true;
  } else {
    // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
    float scale = ratio_ >= 1.0 ? 0.0:1. / (1. - ratio_);
    // mask=true means keep, and mask=false means not keep, so we will
    // generate probability depending on 1-ratio.
    at::bernoulli_distribution<double> dist(1. - ratio_);
    const float* Xdata = X.data<float>();
    float* Ydata = Y->template mutable_data<float>();
    auto mask = Output(1, X.sizes(), at::dtype<bool>());
    bool* mask_data = mask->template mutable_data<bool>();
    auto* gen = context_.RandGenerator();
    for (int i = 0; i < X.numel(); ++i) {
      mask_data[i] = dist(gen) > 0.5;
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      Ydata[i] = Xdata[i] * scale * mask_data[i];
    }
    return true;
  }
}

template <>
bool DropoutGradientOp<float, CPUContext>::RunOnDevice() {
  auto& dY = Input(0);

  auto* dX = Output(0, dY.sizes(), at::dtype<float>());
  if (is_test_) {
    if (dX != &dY) {
      context_.CopyFromCPU<float>(
          dY.numel(), dY.data<float>(), dX->template mutable_data<float>());
    }
    return true;
  } else {
    auto& mask = Input(1);
    CAFFE_ENFORCE_EQ(dY.numel(), mask.numel());
    const float* dYdata = dY.data<float>();
    const bool* mask_data = mask.data<bool>();
    float* dXdata = dX->template mutable_data<float>();
    // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
    float scale = ratio_ >= 1.0 ? 0.0:1. / (1. - ratio_);
    for (int i = 0; i < dY.numel(); ++i) {
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      dXdata[i] = dYdata[i] * mask_data[i] * scale;
    }
    return true;
  }
}

REGISTER_CPU_OPERATOR(Dropout, DropoutOp<float, CPUContext>);
REGISTER_CPU_GRADIENT_OPERATOR(
    DropoutGrad,
    DropoutGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(Dropout)
    .NumInputs(1)
    .NumOutputs(1, 2)
    .AllowInplace({{0, 0}})
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      CAFFE_ENFORCE_EQ(1, in.size());
      vector<TensorShape> out;
      ArgumentHelper argsHelper(def);
      out.push_back(in[0]);
      if (def.output().size() == 2) {
        out.push_back(in[0]);
        out[1].set_data_type(TensorProto_DataType_BOOL);
      }
      return out;
    })
    .SetDoc(R"DOC(

`Dropout` takes one input data tensor (`X`) and produces two tensor outputs, `Y` and
`mask`. If the `is_test` argument is zero (default=0), the output `Y` will be the input
with random elements zeroed. The probability that a given element is zeroed is
determined by the `ratio` argument.

If the `is_test` argument is set to non-zero, the output `Y` is exactly the same as the
input `X`. Note that outputs are scaled by a factor of $\frac{1}{1-ratio}$ during
training, so that during test time, we can simply compute an identity function. This
scaling is important because we want the output at test time to equal the expected value
at training time. Dropout has been proven to be an effective regularization technique to
prevent overfitting during training.


Github Links:

- https://github.com/pytorch/pytorch/blob/main/caffe2/operators/dropout_op.h
- https://github.com/pytorch/pytorch/blob/main/caffe2/operators/dropout_op.cc


<details>

<summary> <b>Example</b> </summary>

**Code**

```
workspace.ResetWorkspace()

op = core.CreateOperator(
    "Dropout",
    ["X"],
    ["Y"] + ["mask"],
    ratio=0.5,
    is_test=0
)

workspace.FeedBlob("X", np.random.randint(10, size=(5, 5)).astype(np.float32))
print("X:", workspace.FetchBlob("X"))
workspace.RunOperatorOnce(op)
print("Y:", workspace.FetchBlob("Y"))
print("mask:", workspace.FetchBlob("mask"))
```

**Result**

```
X: [[5. 4. 3. 6. 9.]
 [2. 1. 8. 0. 9.]
 [7. 3. 0. 6. 3.]
 [1. 8. 2. 6. 4.]
 [6. 2. 6. 4. 0.]]
Y: [[ 0.  0.  0. 12. 18.]
 [ 0.  0. 16.  0.  0.]
 [ 0.  0.  0. 12.  6.]
 [ 0.  0.  4.  0.  0.]
 [12.  0.  0.  0.  0.]]
mask: [[False False False  True  True]
 [False False  True  True False]
 [False False  True  True  True]
 [False False  True False False]
 [ True False False False False]]
```

</details>

)DOC")
    .Arg(
        "ratio",
        "*(type: float; default: 0.5)* Probability of an element to be zeroed.")
    .ArgIsTest(
        "*(type: int; default: 0)* If zero (train mode), perform dropout. If non-zero"
        "(test mode), Y = X.")
    .Input(0, "X", "*(type: Tensor`<float>`)* Input data tensor.")
    .Output(0, "Y", "*(type: Tensor`<float>`)* Output tensor.")
    .Output(
        1,
        "mask",
        "*(type: Tensor`<bool>`)* The output mask containing boolean values for"
        "each element, signifying which elements are dropped out. If `is_test` is"
        "nonzero, this output is not filled.")
    .InheritOnnxSchema();

GRADIENT_OPERATOR_SCHEMA(DropoutGrad)
    .NumInputs(1, 2)
    .NumOutputs(1)
    .AllowInplace({{0, 0}});

class GetDropoutGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    ArgumentHelper argshelper(def_);
    // NOLINTNEXTLINE(modernize-use-bool-literals)
    auto is_test = argshelper.GetSingleArgument<bool>("is_test", 0);
    if (is_test) {
      return SingleGradientDef(
          "DropoutGrad", "", vector<string>{GO(0)}, vector<string>{GI(0)});
    } else {
      return SingleGradientDef(
          "DropoutGrad",
          "",
          vector<string>{GO(0), O(1)},
          vector<string>{GI(0)});
    }
  }
};
REGISTER_GRADIENT(Dropout, GetDropoutGradient);
} // namespace caffe2
