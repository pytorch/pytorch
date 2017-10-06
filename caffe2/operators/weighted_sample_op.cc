#include "caffe2/operators/weighted_sample_op.h"

namespace caffe2 {
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
