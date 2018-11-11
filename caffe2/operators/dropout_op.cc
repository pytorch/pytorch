#include "caffe2/operators/dropout_op.h"

#include <string>
#include <vector>

#include "caffe2/utils/eigen_utils.h"

namespace caffe2 {

template <>
void DropoutOp<float, CPUContext>::DropoutForward(
    const int N,
    const float* X,
    float* Y,
    bool* mask) {
  float* uniform_data = Y;
  if (Y == X) {
    uniform_.Resize(N);
    uniform_data = uniform_.mutable_data<float>();
  }
  EigenVectorArrayMap<bool> mask_arr(mask, N);
  EigenVectorArrayMap<float> uniform_arr(uniform_data, N);
  uniform_arr.setRandom();
  // Change threshold since setRandom() generates uniform distribution in range
  // [-1, 1].
  const float threshold = ratio_ * 2.0f - 1.0f;
  mask_arr = uniform_arr > threshold;
  const float scale = 1.0f / (1.0f - ratio_);
  EigenVectorArrayMap<float>(Y, N) =
      ConstEigenVectorArrayMap<float>(X, N) * mask_arr.cast<float>() * scale;
}

template <>
void DropoutGradientOp<float, CPUContext>::DropoutBackward(
    const int N,
    const float* dY,
    const bool* mask,
    float* dX) {
  const float scale = 1.0f / (1.0f - ratio_);
  EigenVectorArrayMap<float>(dX, N) = ConstEigenVectorArrayMap<float>(dY, N) *
      ConstEigenVectorArrayMap<bool>(mask, N).cast<float>() * scale;
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

- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/dropout_op.h
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/dropout_op.cc


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
    .ArgIsTest(
        "*(type: int; default: 0)* If zero (train mode), perform dropout. If non-zero"
        "(test mode), Y = X.")
    .Arg(
        "ratio",
        "*(type: float; default: 0.5)* Probability of an element to be zeroed.")
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

namespace {

class GetDropoutGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;

  std::vector<OperatorDef> GetGradientDefs() override {
    ArgumentHelper argshelper(def_);
    auto is_test = argshelper.GetSingleArgument<bool>("is_test", 0);
    if (is_test) {
      return SingleGradientDef(
          "DropoutGrad", "", vector<string>{GO(0)}, vector<string>{GI(0)});
    } else {
      return SingleGradientDef(
          "DropoutGrad",
          "",
          std::vector<std::string>{GO(0), O(1)},
          std::vector<std::string>{GI(0)});
    }
  }
};

} // namespace

REGISTER_GRADIENT(Dropout, GetDropoutGradient);

} // namespace caffe2
