#include "caffe2/operators/lengths_tile_op.h"

namespace caffe2 {
namespace {

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

GRADIENT_NOT_IMPLEMENTED_YET(LengthsTile);

} // namespace
} // namespace caffe2
