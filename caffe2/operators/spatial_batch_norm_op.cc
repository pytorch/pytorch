#include "caffe2/operators/spatial_batch_norm_op.h"

#include <array>

#include "caffe2/utils/eigen_utils.h"

namespace caffe2 {

namespace {

OpSchema::Cost CostInferenceForSpatialBN(
    const OperatorDef& def,
    const vector<TensorShape>& in) {
  struct OpSchema::Cost cost = PointwiseCostInference<4>(def, in);
  ArgumentHelper helper(def);
  auto order =
      StringToStorageOrder(helper.GetSingleArgument<string>("order", "NCHW"));
  const TensorShape X = in[0];
  const int C =
      (order == StorageOrder::NCHW ? X.dims(1) : X.dims(X.dims_size() - 1));
  cost.params_bytes = 2 * C * sizeof(float);
  return cost;
}

} // namespace

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(SpatialBN, SpatialBNOp<CPUContext>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(SpatialBN)
    .NumInputs({5, 7})
    .NumOutputs({1, 5})
    .AllowInplace({{0, 0}, {5, 3}, {6, 4}})
    .EnforceInplace({{3, 1}, {4, 2}})
    .CostInferenceFunction(CostInferenceForSpatialBN)
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      ArgumentHelper helper(def);
      bool is_test = helper.GetSingleArgument<int>(OpSchema::Arg_IsTest, 0);

      if (!is_test) {
        vector<TensorShape> out;
        StorageOrder order = StringToStorageOrder(
            helper.GetSingleArgument<string>("order", "NCHW"));
        const TensorShape& X = in[0];
        const int C =
            (order == StorageOrder::NCHW ? X.dims(1)
                                         : X.dims(X.dims_size() - 1));

        out.push_back(in[0]);
        TensorShape meanvar_tp =
            CreateTensorShape(vector<int>{C}, TensorProto::FLOAT);
        out.push_back(meanvar_tp); // RUNNING_MEAN
        out.push_back(meanvar_tp); // RUNNING_MEAN
        out.push_back(meanvar_tp); // SAVED_MEAN
        out.push_back(meanvar_tp); // SAVED_VAR
        return out;
      } else {
        return vector<TensorShape>{in[0]};
      }
    })
    .SetDoc(R"DOC(
Applies spatial batch normalization to the input tensor as described in the original paper, [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167). Be aware, this operator has two different output sets, depending on the value of *is_test*. According to the paper, the primary operation of spatial batch normalization is:

$$Y = \frac{X - \mu_x}{\sqrt{\sigma^2_{x} + \epsilon}}*\gamma + b$$

In the equation, $\mu_x$ is the *mean*, $X$ is the input data, $\sigma^2_{x}$ is the *var*, $\epsilon$ is *epsilon*, $\gamma$ is the *scale*, $b$ is the *bias*, and $Y$ is the output data. The *momentum* arg also affects this calculation in the computation of the running mean and variance. The influence of *momentum* is as follows:

$$running\_mean = running\_mean * momentum + mean * (1 - momentum)$$

$$running\_var = running\_var * momentum + var * (1 - momentum)$$

Output when is_test = 0 (train mode): *Y, mean, var, saved_mean, saved_var*

Output when is_test = 1 (test mode): *Y*

Github Links:
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/spatial_batch_norm_op.cc
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/spatial_batch_norm_op.h

)DOC")
    .ArgIsTest(
        "*(type: int; default: 0)* If set to nonzero, run spatial batch normalization in test mode.")
    .Arg(
        "epsilon",
        "*(type: float; default: 1e-5)* The epsilon value to use to avoid division by zero.")
    .Arg(
        "order",
        // NOLINTNEXTLINE(modernize-raw-string-literal)
        "*(type: string; default: \"NCHW\")* Specifies the order of the input data blob, where $N$ is batch size, $C$ is number of channels, $H$ is spatial height, and $W$ is spatial width. The only other valid option is \"NHWC\".")
    .Arg(
        "momentum",
        "*(type: float; default: 0.9)* Factor used in computing the running mean and variance. e.g., running_mean = running_mean x momentum + mean x (1 - momentum)")
    .Arg(
        "num_batches",
        "*(type: int; default: 1)* Specifies the number of batches to apply normalization on. Requires specifying the optional sums and sumsq inputs that provide statistics across multiple batches from which mean and variance can be determined.")
    .Input(
        0,
        "X",
        "The input 4-dimensional tensor of shape $NCHW$ or $NHWC$ depending on the order parameter.")
    .Input(
        1,
        "scale",
        "The scale as a 1-dimensional tensor of size $C$ to be applied to the output.")
    .Input(
        2,
        "bias",
        "The bias as a 1-dimensional tensor of size $C$ to be applied to the output.")
    .Input(
        3,
        "mean",
        "The running mean (training) or the estimated mean (testing) as a 1-dimensional tensor of size $C$.")
    .Input(
        4,
        "var",
        "The running variance (training) or the estimated variance (testing) as a 1-dimensional tensor of size $C$.")
    .Input(
        5,
        "sums",
        "*(optional)* Per-channel sums of elements to be used to determine the mean and variance for this batch.")
    .Input(
        6,
        "sumsq",
        "*(optional)* Per-channel sum of elements squared per channel to be used to determine the variance for this batch.")

    .Output(0, "Y", "The output 4-dimensional tensor of the same shape as $X$.")
    .Output(
        1,
        "mean",
        "The running mean after the spatial BN operator. Must be in-place with the input *mean*. Should not be used for testing.")
    .Output(
        2,
        "var",
        "The running variance after the spatial BN operator. Must be in-place with the input *var*. Should not be used for testing.")
    .Output(
        3,
        "saved_mean",
        "Saved mean used during training to speed up gradient computation. Should not be used for testing.")
    .Output(
        4,
        "saved_var",
        "Saved variance used during training to speed up gradient computation. Should not be used for testing.")
    .InheritOnnxSchema("BatchNormalization");

} // namespace caffe2
