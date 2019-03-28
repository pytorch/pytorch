#include "caffe2/operators/adjust_batch_op.h"

namespace caffe2 {
REGISTER_CPU_OPERATOR(AdjustBatch, AdjustBatchOp<CPUContext>);
OPERATOR_SCHEMA(AdjustBatch)
    .NumInputs(1, 2)
    .NumOutputs(1, 2)
    .Input(0, "Input", "Input data")
    .Input(1, "RealBatchSizeIn", "[Optional] Real batch size")
    .Output(0, "Output", "Data with Adjusted batch size")
    .Output(1, "RealBatchSizeOut", "[Optional] Real batah size")
    .Arg("max_batch_size", "(*int*): max batch size")
    .SetDoc(R"DOC(
Adjust the batch size of `input` tensor. When we only have 1 input, it will adjust the batch size according to `max_batch_size` argument. In this case, in addition, if it has two outputs, it will record the input batch size and record it to the second output. When we have 2 inputs, it expects the seocnd input contains the batch size to adjust to, and will truncate the input data accordingly.

Github Links:
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/adjust_batch_op.cc

  )DOC");
} // namespace caffe2
