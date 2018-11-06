#include "caffe2/operators/weighted_sample_op.h"

namespace caffe2 {

template <>
bool WeightedSampleOp<float, CPUContext>::RunOnDevice() {
  CAFFE_ENFORCE_EQ(
      InputSize(),
      OutputSize(),
      "The number of tensors of the input and the output must be the same.");
  auto& weights = Input(0);
  int batch_size = weights.size(0);
  int weights_dim = weights.size(1);
  auto* out_idx = Output(0);

  if (batch_size > 0 && weights_dim > 0) {
    cum_mass_.resize(weights_dim);
    const float* mat_weights = weights.template data<float>();
    const float* mat_values = nullptr;
    out_idx->Resize(batch_size, 1);
    int* output_indices = out_idx->template mutable_data<int>();
    float* output_values = nullptr;

    if (InputSize() == 2) {
      auto& values = Input(1);
      CAFFE_ENFORCE_EQ(
          weights.sizes(),
          values.sizes(),
          "The sampling weights tensor and the sampling values tensor must have the same dimensions.");
      mat_values = values.template data<float>();
      auto* out_value = Output(1);
      out_value->Resize(batch_size, 1);
      output_values = out_value->template mutable_data<float>();
    }

    for (int i = 0; i < batch_size; i++) {
      float r;
      int offset = i * weights_dim;

      cum_mass_[0] = mat_weights[offset];
      for (int j = 1; j < weights_dim; j++) {
        cum_mass_[j] = cum_mass_[j - 1] + mat_weights[offset + j];
      }

      math::RandUniform<float, CPUContext>(
          1, 0.0f, cum_mass_[cum_mass_.size() - 1], &r, &context_);
      // Makes the element in cum_mass_ slightly bigger
      // to compensate inaccuracy introduced due to rounding,
      cum_mass_[cum_mass_.size() - 1] += 0.01f;
      auto lb = lower_bound(cum_mass_.begin(), cum_mass_.end(), r);
      CAFFE_ENFORCE(lb != cum_mass_.end(), "Cannot find ", r, " in cum_mass_.");
      output_indices[i] = static_cast<int>(lb - cum_mass_.begin());

      if (output_values) {
        output_values[i] =
            static_cast<float>(mat_values[offset + (lb - cum_mass_.begin())]);
      }
    }
  } else {
    out_idx->Resize(0);
    out_idx->template mutable_data<int>();
    if (OutputSize() == 2) {
      auto* out_value = Output(1);
      out_value->Resize(0);
      out_value->template mutable_data<float>();
    }
  }

  return true;
}

REGISTER_CPU_OPERATOR(WeightedSample, WeightedSampleOp<float, CPUContext>);

OPERATOR_SCHEMA(WeightedSample)
    .NumInputs(1, 2)
    .NumOutputs(1, 2)
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out(2);
      int batch_size = in[0].dims(0);
      out[0] = CreateTensorShape(vector<int>{batch_size}, TensorProto::INT32);
      out[1] = CreateTensorShape(vector<int>{batch_size}, TensorProto::FLOAT);
      return out;
    })
    .SetDoc(R"DOC(
The operator performs sampling based on the input sampling weights for
each batch. All weights must be non-negative numbers.
The input is a 2-D tensor (Tensor) of size (batch_size x weights_dim).
For each batch, an index is randomly sampled from the distribution given by
the weights of the corresponding batch.
The output is a 1-D tensor (Tensor) of size (batch_size x 1) and
contains the index(es) of the sampled output.
)DOC")
    .Input(
        0,
        "sampling_weights",
        "A 2-D Tensor of size (batch_size x weights_dim)."
        "All weights must be non-negative numbers.")
    .Input(
        1,
        "sampling_values",
        "An optional 2-D Tensor of size (batch_size x weights_dim)."
        "Its values correspond to the sampling weights.")
    .Output(
        0,
        "sampled_indexes",
        "The output tensor contains index(es) sampled from distribution given"
        "by the weight vector(s) in the input tensor"
        "The output is a 1-D Tensor of size (batch_size x 1)")
    .Output(
        1,
        "sampled_values",
        "The output tensor contains value(s) selected by the sampled index(es)"
        "It is a 1-D Tensor of size (batch_size x 1)");

SHOULD_NOT_DO_GRADIENT(WeightedSample);
} // namespace caffe2
