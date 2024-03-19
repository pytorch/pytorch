#include "caffe2/operators/reverse_packed_segs_op.h"

namespace caffe2 {
REGISTER_CPU_OPERATOR(ReversePackedSegs, ReversePackedSegsOp<CPUContext>);

OPERATOR_SCHEMA(ReversePackedSegs)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Reverse segments in a 3-D tensor (lengths, segments, embeddings,), leaving
paddings unchanged. This operator is used to reverse input of a recurrent neural
network to make it a BRNN.
  )DOC")
    .Input(0, "data", "a 3-D (lengths, segments, embeddings,) tensor.")
    .Input(1, "lengths", "length of each segment.")
    .Output(
        0,
        "reversed data",
        "a (lengths, segments, embeddings,) tensor with each segment reversed"
        "and paddings unchanged.");

class GetReversePackedSegsGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "ReversePackedSegs",
        "",
        vector<string>{GO(0), I(1)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(ReversePackedSegs, GetReversePackedSegsGradient);
} // namespace caffe2
