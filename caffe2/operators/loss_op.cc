#include "caffe2/operators/loss_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(AveragedLoss, AveragedLoss<float, CPUContext>);
REGISTER_CPU_OPERATOR(AveragedLossGradient,
                      AveragedLossGradient<float, CPUContext>);

OPERATOR_SCHEMA(AveragedLoss)
  .NumInputs(1)
  .NumOutputs(1)
  .ScalarType(TensorProto::FLOAT)
  .SetDoc(R"DOC(
The *AveragedLoss* op takes a single 1-D input tensor *input* and returns a single output float value *output*. The output represents the average of the values in *input*. This op is commonly used for averaging losses, hence the name, however it does not exclusively operate on losses.

Github Links:

- https://github.com/caffe2/caffe2/blob/master/caffe2/operators/loss_op.h
- https://github.com/caffe2/caffe2/blob/master/caffe2/operators/loss_op.cc


<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "AveragedLoss",
    ["input"],
    ["output"],
)

workspace.FeedBlob("input", np.array([8, 10, 12]).astype(np.float32))
print("input:\n", workspace.FetchBlob("input"))

workspace.RunOperatorOnce(op)
print("output: \n", workspace.FetchBlob("output"))

```

**Result**

```

input:
 [ 8. 10. 12.]
output:
 10.0

```

</details>


)DOC")
  .Input(0, "input", "The input data as Tensor")
  .Output(0, "output", "The output tensor of size 1 containing the averaged value.");

OPERATOR_SCHEMA(AveragedLossGradient).NumInputs(2).NumOutputs(1);

class GetAveragedLossGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "AveragedLossGradient", "",
        vector<string>{I(0), GO(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(AveragedLoss, GetAveragedLossGradient);

}  // namespace caffe2
