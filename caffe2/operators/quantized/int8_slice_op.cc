#include "caffe2/operators/quantized/int8_slice_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(Int8Slice, int8::Int8SliceOp);

OPERATOR_SCHEMA(Int8Slice)
    .NumInputs(1, 3)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Produces a slice of the input Int8 tensor. Currently, only slicing in a single
dimension is supported.
Slices are passed as 2 1D vectors or as two keyword argument lists with starting
and end indices for each dimension of the input `data` tensor. If a negative
value is passed for any of the start or end indices, it represents the number of
elements before the end of that dimension. End indices are non-inclusive unless
negative (end index -1 means up to and including the last element).


Example:

  data = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
  ]
  starts = [0, 1]
  ends = [-1, 3]

  result = [
      [2, 3],
      [6, 7],
  ]
)DOC")
    .Input(0, "data", "Int8 Tensor of data to extract slices from.")
    .Input(1, "starts", "1D tensor: start-indices for each dimension of data.")
    .Input(2, "ends", "1D tensor: end-indices for each dimension of data.")
    .Arg("Y_scale", "Output tensor quantization scale")
    .Arg("Y_zero_point", "Output tensor quantization offset")
    .Arg("starts", "List of starting indices")
    .Arg("ends", "List of ending indices")
    .Arg(
        "dim",
        "(Optional) The dimension to slice over. If specified start_idx and end_idx should also be given and it takes precedence over starts and ends")
    .Arg("start_idx", "(Optional) The dimension to start slice from. Default is 0")
    .Arg("end_idx", "(Optional) The dimension to end the slice. Default is -1")
    .Output(0, "output", "Sliced Int8 data tensor.")
    .InheritOnnxSchema("Slice");

} // namespace caffe2
