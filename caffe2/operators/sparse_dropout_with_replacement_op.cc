#include "caffe2/operators/sparse_dropout_with_replacement_op.h"

#include <algorithm>
#include <iterator>

namespace caffe2 {

template <>
bool SparseDropoutWithReplacementOp<CPUContext>::RunOnDevice() {
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
  int32_t total_output_length = 0;
  vector<bool> selected(Lengths.numel(), true);
  for (int i = 0; i < Lengths.numel(); ++i) {
    if (dist(gen) > 0.5) {
      output_lengths_data[i] = input_lengths_data[i];
    } else {
      // Replace with a single dropout value.  Even if input length is 0.
      output_lengths_data[i] = 1;
      selected[i] = false;
    }
    total_output_length += output_lengths_data[i];
  }

  auto* Y = Output(0, {total_output_length}, at::dtype<int64_t>());
  int64_t* Ydata = Y->template mutable_data<int64_t>();

  int input_index = 0;
  int output_index = 0;
  for (int i = 0; i < Lengths.numel(); ++i) {
    if (selected[i]) {
      // Copy logical elements from input to output
      for (int j = input_index; j < input_index + input_lengths_data[i]; ++j) {
        Ydata[output_index++] = Xdata[j];
      }
    } else {
      Ydata[output_index++] = replacement_value_;
    }
    input_index += input_lengths_data[i];
  }
  return true;
}

REGISTER_CPU_OPERATOR(
    SparseDropoutWithReplacement,
    SparseDropoutWithReplacementOp<CPUContext>);

OPERATOR_SCHEMA(SparseDropoutWithReplacement)
    .NumInputs(2)
    .SameNumberOfOutput()
    .SetDoc(R"DOC(

`SparseDropoutWithReplacement` takes a 1-d input tensor and a lengths tensor.
Values in the Lengths tensor represent how many input elements constitute each
example in a given batch.  The set of input values for an example will be
replaced with the single dropout value with probability given by the `ratio`
argument.

<details>

<summary> <b>Example</b> </summary>

**Code**

```
workspace.ResetWorkspace()

op = core.CreateOperator(
    "SparseDropoutWithReplacement",
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

NO_GRADIENT(SparseDropoutWithReplacement);
} // namespace caffe2
