#include "caffe2/operators/bucketize_op.h"

#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {

template <>
bool BucketizeOp<CPUContext>::RunOnDevice() {
  auto& input = Input(X);
  CAFFE_ENFORCE_GE(input.dim(), 1);

  auto N = input.numel();
  auto* output = Output(INDICES, input.sizes(), at::dtype<int32_t>());
  const auto* input_data = input.template data<float>();
  auto* output_data = output->template mutable_data<int32_t>();

  math::Set<int32_t, CPUContext>(output->numel(), 0.0, output_data, &context_);

  for (int64_t pos = 0; pos < N; pos++) {
    // here we assume the boundary values for each feature are sorted
    auto bucket_idx =
        std::lower_bound(
            boundaries_.begin(), boundaries_.end(), input_data[pos]) -
        boundaries_.begin();
    output_data[pos] = bucket_idx;
  }

  return true;
};
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(Bucketize, BucketizeOp<CPUContext>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(Bucketize)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
This operator works as bucketize in tensorflow and digitize
in numpy. It bucketizes the input 'X' based on argument 'boundaries'.
For each value x in input 'data', the operator returns index i given
boundaries[i-1] < x <= boundaries[i].
If values in 'data' are beyond the bounds of boundaries, 0 or
len(boundaries) is returned as appropriate.
The boundaries need to be monotonically increasing.
For example

If data = [2, 4, 1] and boundaries = [0.1, 2.5], then

output = [1, 2, 1]

If data = [[2, 3], [4, 1], [2, 5]] and boundaries = [0.1, 2.5], then

output = [[1, 2], [2, 1], [1, 2]]

)DOC")
    .Input(0, "data", "input tensor")
    .Output(
        0,
        "output",
        "indices of bins given by boundaries to which each value"
        "in data belongs")
    .TensorInferenceFunction([](const OperatorDef& /* def */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out(in);
      out[0].set_data_type(TensorProto::INT32);
      return out;
    })
    .Arg("boundaries", "bucketization boundaries");

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
NO_GRADIENT(BucketizeOp);
} // namespace caffe2

using BucketizeInt = caffe2::BucketizeOp<caffe2::CPUContext>;

C10_EXPORT_CAFFE2_OP_TO_C10_CPU(
    Bucketize,
    "_caffe2::Bucketize(Tensor data, float[] boundaries) -> Tensor output",
    BucketizeInt);
