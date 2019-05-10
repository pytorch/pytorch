#include "caffe2/operators/quantized/int8_concat_op.h"

#include "caffe2/operators/concat_split_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(Int8Concat, int8::Int8ConcatOp);

OPERATOR_SCHEMA(Int8Concat)
    .NumInputs(1, INT_MAX)
    .NumOutputs(1, 2)
    .Arg("Y_scale", "Output tensor quantization scale")
    .Arg("Y_zero_point", "Output tensor quantization offset")
    .Arg("axis", "Which axis to concat on")
    .Arg(
        "add_axis",
        "Pass 1 to add the axis specified in arg 'axis' to all "
        "input tensors")
    .TensorInferenceFunction(
        OpSchema::NeedsAllInputShapes(TensorInferenceForConcat))
    .CostInferenceFunction(CostInferenceForConcat)
    .SetDoc("Concatenate a list of tensors into a single tensor")
    .Output(0, "concat_result", "Concatenated tensor")
    .Output(1, "split_info", "The dimensions of the inputs.")
    .InheritOnnxSchema("Concat");

} // namespace caffe2
