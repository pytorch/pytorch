#include "caffe2/operators/spatial_batch_norm_op.h"

namespace caffe2 {

namespace {
// Input: X, scale, bias (if training mode)
// Input: X, scale, bias, estimated_mean, estimated_inv_variance
//     (if inference mode)
// Output: Y, running_mean, running_inv_variance (if training mode, type 1)
// Output: Y, running_mean, running_inv_variance, saved_mean,
//         saved_inv_variance (if training mode, type 2)
// Output: Y (if inference mode)
OPERATOR_SCHEMA(SpatialBN)
    .NumInputs({3, 5}).NumOutputs({1, 3, 5});
// Input: X, scale, dY  (type 1)
// Input: X, scale, dY, saved_mean, saved_inv_variance
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
      CAFFE_CHECK(arg.has_i());
      is_test = arg.i();
    }
    vector<string> grad_outputs{GI(0), GI(1), GI(2)};
    vector<string> grad_inputs;
    if (is_test) {
      // This is in testing mode. The operator should have five input:
      //     X, scale, bias, estimated_mean, estimated_inv_variance
      // The gradient inputs are:
      //     X, scale, dY, estimated_mean, estimated_inv_variance
      CAFFE_CHECK_EQ(def_.input_size(), 5);
      CAFFE_CHECK_EQ(def_.output_size(), 1);
      grad_inputs = vector<string>{
          I(0), I(1), GO(0), I(3), I(4)};
    } else {
      CAFFE_CHECK_EQ(def_.input_size(), 3);
      CAFFE_CHECK(def_.output_size() == 3 || def_.output_size() == 5);
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
        CAFFE_LOG_FATAL << "Should not happen.";
      }
    }
    return SingleGradientDef(
        "SpatialBNGradient", "", grad_inputs, grad_outputs);
  }
};
REGISTER_GRADIENT(SpatialBN, GetSpatialBNGradient);
}  // namespace

}  // namespace caffe2
