#include "caffe2/operators/fused_rowwise_random_quantization_ops.h"
#include "caffe2/core/registry.h"

namespace caffe2 {
REGISTER_CPU_OPERATOR(
    FloatToFusedRandRowwiseQuantized,
    FloatToFusedRandRowwiseQuantizedOp<CPUContext>);
OPERATOR_SCHEMA(FloatToFusedRandRowwiseQuantized)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      ArgumentHelper helper(def);
      auto bitwidth = helper.GetSingleArgument<int32_t>("bitwidth", 8);
      size_t data_per_byte = 8 / bitwidth;
      vector<TensorShape> out;
      TensorShape X = in[0];
      X.set_dims(1, 10 + (X.dims(1) + data_per_byte - 1) / data_per_byte);
      out.push_back(std::move(X));
      out[0].set_data_type(TensorProto_DataType_UINT8);
      return out;
    })
    .SetDoc(R"DOC(
Applies row-wise stochastic/random quantization by determining the range of
each row in the input matrix, and then quantize each element to one of two
closest discrete levels by randomly drawing Bernoulli distribution.
The method is extended from TernGrad [1],
which randomly quantizes gradients to three levels to reduce communication in distributed training.
The format of each row (x) in the output matrix is [bitwidth][tail][min][max][data]:
bitwidth[1 Byte]: bitwidth per data [1, 2, 4 or 8];
tail[1 Byte]: the number of unused buckets [1-8] (One byte is split to 8/bitwidth buckets and each bucket stores one low-precision data in bitwidth bits);
min[4 Bytes]: the minimum floating value min(x);
max[4 Bytes]: the maximum floating value max(x);
data: quantized data.
The quantization is uniform with levels q = min + (max-min)/(2^bitwidth - 1)*[0:1:2^bitwidth].
During stochastic/random quantization x'=Quantize(x), for q_j < x_i <= q_{j+1}, we draw quantization x'_i from Bernoulli distributions with
P(x'_i = q_{j+1}) = (x_i - q_j)/(q_{j+1} - q_j), and
P(x'_i = q_j) = (q_{j+1} - x_i)/(q_{j+1} - q_j) where x'_i is the quantized value of x_i.
[1] proved E{x'_i}=x_i, which is an unbiased approximation. More details are in the paper.
For example, suppose targeted bitwidth = 2 and x = [0.3, -1.4, -0.6, 0.9, 1.0],
then tail = 3, min = -1.4, max = 1.0 and q = [-1.4, -0.6, 0.2, 1.0].
x_1 = 0.3 will be quantized to x'_1 = 0.2 with probability 7/8 and to x'_1 = 1.0 with probability 1/8.
The storage format of quantized data is: [x'_1|x'_3|x'_5|xxx]-[x'_2|x'_4|xxx|xxx].
In general, a input row is split to multiple segments. One segment is a continuous subarray of the row,
and its length is the number of bytes storing quantized data in the output matrix.
The b-th bucket of the i-th byte stores the i-th data of the b-th segment of input row.

[1] Wen, Wei, Cong Xu, Feng Yan, Chunpeng Wu, Yandan Wang, Yiran Chen, and Hai Li.
"Terngrad: Ternary gradients to reduce communication in distributed deep learning."
In Advances in Neural Information Processing Systems, pp. 1508-1518. 2017.

)DOC")
    .Input(0, "input", "Float32 input data")
    .Output(0, "output", "Fused bitwidth, tail, min, max and quantized data")
    .Arg("bitwidth", "How many bits to quantiz per data (defaults to 8).")
    .Arg("random", "random or not (True). False is set up for unittest.");
NO_GRADIENT(FloatToFusedRandRowwiseQuantized);

REGISTER_CPU_OPERATOR(
    FusedRandRowwiseQuantizedToFloat,
    FusedRandRowwiseQuantizedToFloatOp<CPUContext>);
OPERATOR_SCHEMA(FusedRandRowwiseQuantizedToFloat)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>&) {
      vector<TensorShape> out;
      for (int i = 0; i < def.output_size(); i++) {
        TensorShape ts;
        ts.set_unknown_shape(true);
        ts.set_data_type(TensorProto_DataType_FLOAT);
        out.push_back(ts);
      }
      return out;
    })
    .SetDoc(R"DOC(
De-quantizes the result of the FloatToFusedRandRowwiseQuantized operator.
Refer FloatToFusedRandRowwiseQuantized operator for details.
)DOC")
    .Input(
        0,
        "quantized_input",
        "Fused bitwidth, tail, min, max and quantized data")
    .Output(0, "float_input", "Float32 data");
NO_GRADIENT(FusedRandRowwiseQuantizedToFloat);
} // namespace caffe2
