#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/tile_op.h"

namespace caffe2 {
namespace {
REGISTER_CUDA_OPERATOR(Tile, TileOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(TileGradient, TileGradientOp<float, CUDAContext>);
} // namespace
} // namespace caffe2
