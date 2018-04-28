#include "caffe2/operators/numpy_tile_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(NumpyTile, NumpyTileOp<CPUContext>);

OPERATOR_SCHEMA(NumpyTile)
    .NumInputs(2)
    .Input(0, "data", "The input tensor.")
    .Input(1, "repeats", "1-D Tensor specifying how many times to repeat"
                         " each axis.")
    .Output(
        0,
        "tiled_data",
        "Tensor that will contain input replicated along the given axis.")
    .InheritOnnxSchema("Tile");

} // namespace caffe2
