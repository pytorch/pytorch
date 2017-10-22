#include "caffe2/operators/weighted_sample_op.h"

namespace caffe2 {

template <>
bool WeightedSampleOp<float, CPUContext>::RunOnDevice() {
  auto& weights = Input(0);
  int batch_size = weights.dim(0);
  int weights_dim = weights.dim(1);
  auto* output = Output(0);

  if (batch_size > 0 && weights_dim > 0) {
    output->Resize(batch_size, 1);
    cum_mass_.resize(weights_dim);
    const float* mat_weights = weights.template data<float>();
    int* output_indices = output->template mutable_data<int>();

    for (int i = 0; i < batch_size; i++) {
      offset_ = i * weights_dim;

      cum_mass_[0] = mat_weights[offset_];
      for (int j = 1; j < weights_dim; j++) {
        cum_mass_[j] = cum_mass_[j - 1] + mat_weights[offset_ + j];
      }

      math::RandUniform<float, CPUContext>(
          1, 0.0f, cum_mass_[cum_mass_.size() - 1], &r_, &context_);
      // Makes the element in cum_mass_ slightly bigger
      // to compensate inaccuracy introduced due to rounding,
      cum_mass_[cum_mass_.size() - 1] += 0.01f;
      auto lb = lower_bound(cum_mass_.begin(), cum_mass_.end(), r_);
      CAFFE_ENFORCE(
          lb != cum_mass_.end(), "Cannot find ", r_, " in cum_mass_.");
      output_indices[i] = static_cast<int>(lb - cum_mass_.begin());
    }
  } else {
    output->Resize(0);
    output->template mutable_data<int>();
  }

  return true;
}

REGISTER_CPU_OPERATOR(WeightedSample, WeightedSampleOp<float, CPUContext>);

OPERATOR_SCHEMA(WeightedSample)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out(1);
      int batch_size = in[0].dims(0);
      out[0] = CreateTensorShape(vector<int>{batch_size}, TensorProto::INT32);
      return out;
    })
    .SetDoc(R"DOC(

    The operator performs sampling based on the input sampling weights for each batch.
    All weights must be non-negative numbers.
    The input is a 2-D tensor (Tensor<float>) of size (batch_size x weights_dim).
    For each batch, an index is randomly sampled from the distribution given by the weights of the corresponding batch.
    The output is a 1-D tensor (Tensor<int>) of size (batch_size x 1) and contains the index(es) of the sampled output.


    )DOC")
    .Input(
        0,
        "sampling_weights",
        "The input is a 2-D tensor (Tensor<float>) of size (batch_size x weights_dim)."
        "All weights must be non-negative numbers.")
    .Output(
        0,
        "sampled_indexes",
        "The output tensor contains index(es) sampled from distribution given by the weight vector(s) in the input tensor"
        "The output is a 1-D tensor (Tensor<int>) of size (batch_size x 1)");

SHOULD_NOT_DO_GRADIENT(WeightedSample);
} // namespace caffe2
