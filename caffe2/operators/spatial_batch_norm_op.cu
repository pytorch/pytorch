#include "caffe2/operators/spatial_batch_norm_op.h"

#include "caffe2/operators/spatial_batch_norm_op_impl.cuh"

namespace caffe2 {

REGISTER_CUDA_OPERATOR(SpatialBN, SpatialBNOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(SpatialBNGradient, SpatialBNGradientOp<CUDAContext>);

} // namespace caffe2
