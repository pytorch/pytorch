#include "caffe2/operators/quantized/int8_channel_shuffle_op.h"

namespace caffe2 {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(Int8ChannelShuffle, int8::Int8ChannelShuffleOp);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(Int8ChannelShuffle)
    .IdenticalTypeAndShape()
    .Arg("Y_scale", "Output tensor quantization scale")
    .Arg("Y_zero_point", "Output tensor quantization offset")
    .NumInputs(1)
    .NumOutputs(1);

} // namespace caffe2
