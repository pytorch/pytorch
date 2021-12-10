#include "caffe2/operators/weighted_multi_sampling_op.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
bool WeightedMultiSamplingOp<Context>::RunOnDevice() {
  const auto& weight = Input(0);
  CAFFE_ENFORCE_EQ(weight.dim(), 1, "Input should be 1-D vector");
  auto dims = weight.sizes().vec();
  size_t data_size = weight.dim32(0);

  std::vector<int64_t> indices_sizes;
  auto num_samples = num_samples_;
  if (InputSize() == 2) {
    CAFFE_ENFORCE(
        !OperatorBase::HasArgument("num_samples"),
        "New shape is specified by the input blob, do not pass in "
        "the argument `num_samples`.");
    num_samples = Input(1).numel();
    indices_sizes = Input(1).sizes().vec();
  } else {
    indices_sizes = {num_samples};
  }

  auto* indices = Output(0, indices_sizes, at::dtype<int>());
  int* indices_data = indices->template mutable_data<int>();
  if (data_size == 0) {
    indices->Resize(0);
    return true;
  }

  const float* weight_data = weight.template data<float>();

  for (int i = 0; i < num_samples; ++i) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    float r;
    math::RandUniform<float, Context>(
        1, 0.0f, weight_data[data_size - 1], &r, &context_);
    auto lb = std::lower_bound(weight_data, weight_data + data_size, r);
    CAFFE_ENFORCE(
        lb != weight_data + data_size, "Cannot find ", r, " in input CDF.");
    indices_data[i] = static_cast<int>(lb - weight_data);
  }
  return true;
}

REGISTER_CPU_OPERATOR(
    WeightedMultiSampling,
    WeightedMultiSamplingOp<CPUContext>);

OPERATOR_SCHEMA(WeightedMultiSampling)
    .NumInputs(1, 2)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out(1);
      if (in[0].dims(0) == 0) {
        out[0].set_data_type(TensorProto::INT32);
        out[0].add_dims(0);
        return out;
      }

      const ArgumentHelper args(def);
      if (args.HasArgument("num_samples")) {
        CAFFE_ENFORCE_EQ(
            in.size(),
            1,
            "New shape must not be specified by the input blob and the "
            "argument `num_samples` at the same time.");
        int num_samples = args.GetSingleArgument<int64_t>("num_samples", 0);
        out[0] =
            CreateTensorShape(vector<int64_t>{num_samples}, TensorProto::INT32);
        return out;
      } else {
        CAFFE_ENFORCE_EQ(
            in.size(),
            2,
            "New shape must be specified by either the input blob or the "
            "argument `num_samples`.");
        std::vector<int64_t> output_dims = GetDimsVector(in[1]);
        out[0] = CreateTensorShape(output_dims, TensorProto::INT32);
        return out;
      }
    })
    .SetDoc(R"DOC(
The operator performs sampling based on the input sampling weights.
All weights are cummulative probability thus sorted. The output is
a 1-D tensor (Tensor). If two inputs are given, the second input
is used to provide shape of the output sample tensor. Otherwise, we use
argument `num_samples` to determine the number of samples to generate.
)DOC")
    .Input(
        0,
        "sampling_cdf",
        "An optional 1-D Tensor."
        "Input cumulative sampling probability (such as [0.2, 0.5, 0.8, 1.5])."
        " All weights must be non-negative numbers. Note that the last value of"
        " CDF is not necessary 1. If the last value is not 1, all values in"
        " sampling_cdf will be scaled by this number.")
    .Input(
        1,
        "shape_tensor (optional)",
        "Tensor whose shape will be applied to output.")
    .Output(
        0,
        "sampled_indexes",
        "The output tensor contains indices sampled from distribution given"
        "by the weight vector in the input tensor"
        "The output is a 1-D Tensor of size determined by argument"
        "`num_samples` or the second input tensor.")
    .Arg("num_samples", "number of samples to sample from the input data");

SHOULD_NOT_DO_GRADIENT(WeightedMultiSample);
} // namespace caffe2
