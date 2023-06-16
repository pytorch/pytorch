#include "caffe2/operators/sparse_itemwise_dropout_with_replacement_op.h"

#include <algorithm>
#include <iterator>

namespace caffe2 {

template <>
bool SparseItemwiseDropoutWithReplacementOp<CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  CAFFE_ENFORCE_EQ(X.ndim(), 1, "Input tensor should be 1-D");
  const int64_t* Xdata = X.data<int64_t>();
  auto& Lengths = Input(1);
  CAFFE_ENFORCE_EQ(Lengths.ndim(), 1, "Lengths tensor should be 1-D");
  auto* OutputLengths = Output(1, Lengths.size(), at::dtype<int32_t>());
  int32_t const* input_lengths_data = Lengths.template data<int32_t>();
  int32_t* output_lengths_data =
      OutputLengths->template mutable_data<int32_t>();
  // Check that input lengths add up to the length of input data
  int total_input_length = 0;
  for (int i = 0; i < Lengths.numel(); ++i) {
    total_input_length += input_lengths_data[i];
  }
  CAFFE_ENFORCE_EQ(
      total_input_length,
      X.numel(),
      "Inconsistent input data. Number of elements should match total length.");

  at::bernoulli_distribution<double> dist(1. - ratio_);
  auto* gen = context_.RandGenerator();
  const float _BARNUM = 0.5;
  vector<bool> selected(total_input_length, false);
  for (int i = 0; i < total_input_length; ++i) {
    if (dist(gen) > _BARNUM) {
      selected[i] = true;
    }
  }

  for (int i = 0; i < Lengths.numel(); ++i) {
    output_lengths_data[i] = input_lengths_data[i];
  }

  auto* Y = Output(0, {total_input_length}, at::dtype<int64_t>());
  int64_t* Ydata = Y->template mutable_data<int64_t>();

  for (int i = 0; i < total_input_length; ++i) {
    if (selected[i]) {
      // Copy logical elements from input to output
      Ydata[i] = Xdata[i];
    } else {
      Ydata[i] = replacement_value_;
    }
  }
  return true;
}

REGISTER_CPU_OPERATOR(
    SparseItemwiseDropoutWithReplacement,
    SparseItemwiseDropoutWithReplacementOp<CPUContext>);

OPERATOR_SCHEMA(SparseItemwiseDropoutWithReplacement)
    .NumInputs(2)
    .SameNumberOfOutput()
    .SetDoc(R"DOC(

`SparseItemwiseDropoutWithReplacement` takes a 1-d input tensor and a lengths tensor.
Values in the Lengths tensor represent how many input elements constitute each
example in a given batch.  The each input value in the tensor of an example can be
replaced with the replacement value with probability given by the `ratio`
argument.

<details>

<summary> <b>Example</b> </summary>

**Code**

```
workspace.ResetWorkspace()

op = core.CreateOperator(
    "SparseItemwiseDropoutWithReplacement",
    ["X", "Lengths"],
    ["Y", "OutputLengths"],
    ratio=0.5,
    replacement_value=-1
)

workspace.FeedBlob("X", np.array([1, 2, 3, 4, 5]).astype(np.int64))
workspace.FeedBlob("Lengths", np.array([2, 3]).astype(np.int32))
print("X:", workspace.FetchBlob("X"))
print("Lengths:", workspace.FetchBlob("Lengths"))
workspace.RunOperatorOnce(op)
print("Y:", workspace.FetchBlob("Y"))
print("OutputLengths:", workspace.FetchBlob("OutputLengths"))
```

**Result**

```
X: [1, 2, 3, 4, 5]
Lengths: [2, 3]
Y: [1, 2, -1]
OutputLengths: [2, 1]
```

</details>

)DOC")
    .Arg(
        "ratio",
        "*(type: float; default: 0.0)* Probability of an element to be replaced.")
    .Arg(
        "replacement_value",
        "*(type: int64_t; default: 0)* Value elements are replaced with.")
    .Input(0, "X", "*(type: Tensor`<int64_t>`)* Input data tensor.")
    .Input(
        1,
        "Lengths",
        "*(type: Tensor`<int32_t>`)* Lengths tensor for input.")
    .Output(0, "Y", "*(type: Tensor`<int64_t>`)* Output tensor.")
    .Output(1, "OutputLengths", "*(type: Tensor`<int32_t>`)* Output tensor.");

NO_GRADIENT(SparseItemwiseDropoutWithReplacement);
} // namespace caffe2
