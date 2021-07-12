#include "caffe2/operators/gather_fused_8bit_rowwise_op.h"

namespace caffe2 {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(GatherFused8BitRowwise)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Perform the same operation as Gather, but operating on 8-bit rowwise quantized
matrices with fused storage (where each row stores quantized values, and then
the scale and offset).
DATA needs to have rank 2 and INDICES needs to have rank 1.
)DOC")
    .Input(
        0,
        "DATA",
        "uint8 tensor with rank 2 obtained with operator FloatToFused8BitRowwiseQuantized")
    .Input(
        1,
        "INDICES",
        "Integer vector containing indices of the first dimension of DATA for"
        "the rows that are being gathered")
    .Output(0, "OUTPUT", "output")
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out(1);
      for (auto d : in[1].dims()) {
        out[0].add_dims(d);
      }
      for (int i = 1; i < in[0].dims_size(); ++i) {
        out[0].add_dims(in[0].dims(i));
      }
      out[0].set_data_type(in[0].data_type());
      return out;
    });

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    GatherFused8BitRowwise,
    GatherFused8BitRowwiseOp<CPUContext>);

} // namespace caffe2
