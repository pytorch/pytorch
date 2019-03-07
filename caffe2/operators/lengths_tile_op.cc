#include "caffe2/operators/lengths_tile_op.h"

namespace caffe2 {

template <>
bool LengthsTileOp<CPUContext>::RunOnDevice() {
  auto& data = Input(DATA);
  auto& lengths = Input(LENGTHS);
  auto* output = Output(0);

  CAFFE_ENFORCE_EQ(lengths.dim(), 1, "LENGTHS must be 1-D");
  CAFFE_ENFORCE_GE(data.dim(), 1, "DATA should be at least 1-D");
  CAFFE_ENFORCE_EQ(lengths.numel(), data.size(0));

  // Context::CopyFrom and math::Sum need the same context to avoid race
  // conditions
  // why? CPUContext is not used in Sum
  lengths_host_.CopyFrom(lengths); // sync copy
  auto lengths_size = lengths_host_.numel();
  auto* lengths_data = lengths_host_.data<int32_t>();

  int32_t total_length = 0;
  CPUContext cpuContext;
  math::Sum<int32_t, CPUContext>(
      lengths_size, lengths_data, &total_length, &cpuContext);

  auto shape = data.sizes().vec();
  shape[0] = total_length;
  output->Resize(shape);

  auto block_bytesize = data.size_from_dim(1) * data.dtype().itemsize();
  auto src = static_cast<const char*>(data.raw_data());
  auto out = static_cast<char*>(output->raw_mutable_data(data.dtype()));

  for (int64_t i = 0; i < lengths_size; ++i) {
    auto length = lengths_data[i];
    CAFFE_ENFORCE_GE(length, 0);
    for (int32_t j = 0; j < length; ++j) {
      context_.CopyBytesSameDevice(block_bytesize, src, out);
      out += block_bytesize;
    }
    src += block_bytesize;
  }
  return true;
}

REGISTER_CPU_OPERATOR(LengthsTile, LengthsTileOp<CPUContext>);

OPERATOR_SCHEMA(LengthsTile)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Given DATA tensor of rank r >= 1, and LENGTHS tensor of rank 1, duplicate each
entry of the outer-most dimension of DATA according to LENGTHS, and concatenate
them in an output tensor of rank r.

Example:
  DATA  = [
      [1.0, 1.2],
      [2.3, 3.4],
      [4.5, 5.7],
      [6.8, 7.9],
  ]
  LENGTHS = [0, 1, 3, 2]
  OUTPUT = [
      [2.3, 3.4],
      [4.5, 5.7],
      [4.5, 5.7],
      [4.5, 5.7],
      [6.8, 7.9],
      [6.8, 7.9],
  ]
)DOC")
    .Input(
        0,
        "DATA",
        "Tensor of rank r >= 1. First dimension must be equal to the size of "
        "lengths")
    .Input(1, "LENGTHS", "Tensor of int32 lengths of rank 1")
    .Output(0, "OUTPUT", "Tensor of rank r");

class GetLengthsTileGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    CAFFE_ENFORCE_EQ(def_.input_size(), 2);
    return SingleGradientDef(
        "LengthsSum",
        "",
        // input 1 is the lengths used to repeat
        // DATA in the forward pass
        vector<string>{GO(0), I(1)},
        // only concerned with the gradient on "DATA"
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(LengthsTile, GetLengthsTileGradient);
} // namespace caffe2
