#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/locally_connected_op.h"
#include "caffe2/operators/locally_connected_op_impl.h"

namespace caffe2 {

REGISTER_CUDA_OPERATOR(LC, LocallyConnectedOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    LCGradient,
    LocallyConnectedGradientOp<float, CUDAContext>);

REGISTER_CUDA_OPERATOR(LC1D, LocallyConnectedOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    LC1DGradient,
    LocallyConnectedGradientOp<float, CUDAContext>);

REGISTER_CUDA_OPERATOR(LC2D, LocallyConnectedOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    LC2DGradient,
    LocallyConnectedGradientOp<float, CUDAContext>);

REGISTER_CUDA_OPERATOR(LC3D, LocallyConnectedOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    LC3DGradient,
    LocallyConnectedGradientOp<float, CUDAContext>);

} // namespace caffe2
