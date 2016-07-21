#include "caffe2/operators/spatial_batch_norm_op.h"

namespace caffe2 {

namespace {
// TODO(jiayq): remove the 3-input case to keep strong state-less assumption.
OPERATOR_SCHEMA(SpatialBN)
    .NumInputs(3, 5)
    .NumOutputs({1, 3, 5})
    .EnforceInplace({{3, 1}, {4, 2}})
    .SetDoc(R"DOC(
Carries out spatial batch normalization as described in the paper
https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,
there are multiple cases for the number of outputs, which we list below:

Output case #1: Y, mean, inv_var (if training mode, type 1)
Output case #2: Y, mean, inv_var, saved_mean, saved_inv_var
                (if training mode, type 2)
Output case #3: Y (test mode only)

For training mode, type 2 is faster in the sense that for the backward
pass, it is able to reuse the saved mean and inv_var in the gradient
computation.
)DOC")
    .Arg(
        "is_test",
        "If set to nonzero, run spatial batch normalization in test mode.")
    .Arg("epsilon", "The epsilon value to use to avoid division by zero.")
    .Arg("order", "A StorageOrder string.")
    .Input(
        0,
        "X",
        "The input 4-dimensional tensor of shape NCHW or NHWC depending "
        "on the order parameter.")
    .Input(
        1,
        "scale",
        "The scale as a tensor of size 1 to be applied to the output.")
    .Input(
        2,
        "bias",
        "The bias as a tensor of size 1 to be applied to the output.")
    .Input(
        3,
        "mean",
        "The running mean (training) or the estimated mean (testing) "
        "as a 1-dimensional vector of size C.")
    .Input(
        4,
        "inv_var",
        "The running inverse variance (training) or the estimated inverse "
        "variance (testing) as a 1-dimensional vector of size C.")
    .Output(0, "Y", "The output 4-dimensional tensor of the same shape as X.")
    .Output(
        1,
        "mean",
        "The running mean after the spatial BN operator. Must be in-place "
        "with the input mean. Should not be used for testing.")
    .Output(
        2,
        "inv_var",
        "The running inverse variance after the spatial BN operator. Must be "
        "in-place with the input inv_var. Should not be used for testing.")
    .Output(
        3,
        "saved_mean",
        "Optional saved mean used during training to speed up gradient "
        "computation. Should not be used for testing.")
    .Output(
        4,
        "saved_inv_var",
        "Optional saved inverse variance used during training to speed up "
        "gradient computation. Should not be used for testing.");

// Input: X, scale, dY  (type 1)
// Input: X, scale, dY, mean, inv_variance
//     (type 2, faster, and also necessary if one wants to compute gradient
//      in testing mode)
// Output: dX, dscale, dbias
OPERATOR_SCHEMA(SpatialBNGradient).NumInputs({3, 5}).NumOutputs(3);
}  // namespace

// TODO: implement the CPU version of spatial batch normalization.

// Spatial batch normalization's gradient, depending on the various input sizes,
// is a bit more complex than usual gradient operators.
namespace {
class GetSpatialBNGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    // Check if we are in training or testing mode.
    bool is_test = false;
    if (HasArgument(def_, "is_test")) {
      const auto& arg = GetArgument(def_, "is_test");
      CHECK(arg.has_i());
      is_test = arg.i();
    }
    vector<string> grad_outputs{GI(0), GI(1), GI(2)};
    vector<string> grad_inputs;
    if (is_test) {
      // This is in testing mode. The operator should have five input:
      //     X, scale, bias, estimated_mean, estimated_inv_variance
      // The gradient inputs are:
      //     X, scale, dY, estimated_mean, estimated_inv_variance
      CHECK_EQ(def_.input_size(), 5);
      CHECK_EQ(def_.output_size(), 1);
      grad_inputs = vector<string>{
          I(0), I(1), GO(0), I(3), I(4)};
    } else {
      CHECK_EQ(def_.input_size(), 3);
      CHECK(def_.output_size() == 3 || def_.output_size() == 5);
      // This is in training mode. The operator should have either three output:
      //     Y, running_mean, running_inv_variance
      // or five:
      //     Y, running_mean, running_inv_variance, saved_mean,
      //     saved_inv_variance
      switch (def_.output_size()) {
      case 3:
        // The original operator does not have saved mean and inv variance,
        // so the gradient operator cannot take advantage of that.
        // The gradient inputs are:
        //     X, scale, dY
        grad_inputs = vector<string>{I(0), I(1), GO(0)};
        break;
      case 5:
        // The original operator does have saved mean and inv variance,
        // and the gradient operator can take advantage of that.
        // The gradient inputs are:
        //     X, scale, dY, saved_mean, saved_inv_variance
        grad_inputs = vector<string>{
            I(0), I(1), GO(0), O(3), O(4)};
        break;
      default:
        LOG(FATAL) << "Should not happen.";
      }
    }
    return SingleGradientDef(
        "SpatialBNGradient", "", grad_inputs, grad_outputs);
  }
};
REGISTER_GRADIENT(SpatialBN, GetSpatialBNGradient);
}  // namespace

}  // namespace caffe2
