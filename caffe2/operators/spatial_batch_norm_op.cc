#include "caffe2/operators/spatial_batch_norm_op.h"

namespace caffe2 {

// TODO: implement the CPU version of spatial batch normalization.

// Spatial batch normalization's gradient, depending on the various input sizes,
// is a bit more complex than usual gradient operators.
namespace {
struct GetSpatialBNGradient : public GetGradientDefBase {
  vector<OperatorDef>* Create(const OperatorDef& def) override {
    // Check if we are in training or testing mode.
    bool is_test = false;
    if (HasArgument(def, "is_test")) {
      const auto& arg = GetArgument(def, "is_test");
      CAFFE_CHECK(arg.has_i());
      is_test = arg.i();
    }
    vector<string> grad_outputs{GI(def, 0), GI(def, 1), GI(def, 2)};
    vector<string> grad_inputs;
    if (is_test) {
      // This is in testing mode. The operator should have five input:
      //     X, scale, bias, estimated_mean, estimated_inv_variance
      // The gradient inputs are:
      //     X, scale, dY, estimated_mean, estimated_inv_variance
      CAFFE_CHECK_EQ(def.input_size(), 5);
      CAFFE_CHECK_EQ(def.output_size(), 1);
      grad_inputs = vector<string>{
          I(def, 0), I(def, 1), GO(def, 0), I(def, 3), I(def, 4)};
    } else {
      CAFFE_CHECK_EQ(def.input_size(), 3);
      CAFFE_CHECK(def.output_size() == 3 || def.output_size() == 5);
      // This is in training mode. The operator should have either three output:
      //     Y, running_mean, running_inv_variance
      // or five:
      //     Y, running_mean, running_inv_variance, saved_mean,
      //     saved_inv_variance
      switch (def.output_size()) {
      case 3:
        // The original operator does not have saved mean and inv variance,
        // so the gradient operator cannot take advantage of that.
        // The gradient inputs are:
        //     X, scale, dY
        grad_inputs = vector<string>{I(def, 0), I(def, 1), GO(def, 0)};
        break;
      case 5:
        // The original operator does have saved mean and inv variance,
        // and the gradient operator can take advantage of that.
        // The gradient inputs are:
        //     X, scale, dY, saved_mean, saved_inv_variance
        grad_inputs = vector<string>{
            I(def, 0), I(def, 1), GO(def, 0), O(def, 3), O(def, 4)};
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