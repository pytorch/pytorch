#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/lengths_tile_op.h"

namespace caffe2 {
REGISTER_CUDA_OPERATOR(LengthsTile, LengthsTileOp<CUDAContext>);
} // namespace caffe2
