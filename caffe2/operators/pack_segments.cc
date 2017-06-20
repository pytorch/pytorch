#include "caffe2/operators/pack_segments.h"

namespace caffe2 {

namespace {

REGISTER_CPU_OPERATOR(PackSegments, PackSegmentsOp<CPUContext>);
REGISTER_CPU_OPERATOR(UnpackSegments, UnpackSegmentsOp<CPUContext>);

OPERATOR_SCHEMA(PackSegments)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(
        "Map N dim tensor to N+1 dim based on length blob. Sequences that \
    are shorter than the longest sequence are padded with zeros.")
    .Input(
        0,
        "lengths",
        "1-d int/long tensor contains the length in each of the output.")
    .Input(1, "tensor", "N dim Tensor.")
    .Output(
        0,
        "packed_tensor",
        "N + 1 dim Tesor"
        "where dim(1) is the max length"
        ", dim(0) is the batch size.")
    .Arg(
        "pad_minf", "Padding number in the packed segments. Use true to pad \
    -infinity, otherwise pad zeros");
OPERATOR_SCHEMA(UnpackSegments)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc("Map N+1 dim tensor to N dim based on length blob")
    .Input(
        0,
        "lengths",
        "1-d int/long tensor contains the length in each of the input.")
    .Input(1, "tensor", "N+1 dim Tensor.")
    .Output(0, "packed_tensor", "N dim Tesor");

class GetPackSegmentsGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "UnpackSegments",
        "",
        vector<string>{I(0), GO(0)},
        vector<string>{GI(1)});
  }
};
REGISTER_GRADIENT(PackSegments, GetPackSegmentsGradient);

class GetUnpackSegmentsGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "PackSegments", "", vector<string>{I(0), GO(0)}, vector<string>{GI(1)});
  }
};
REGISTER_GRADIENT(UnpackSegments, GetUnpackSegmentsGradient);
}
} // namespace
