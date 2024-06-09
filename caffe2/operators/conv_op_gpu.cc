#include "caffe2/operators/conv_op.h"
#include "caffe2/operators/conv_op_impl.h"
#include "caffe2/core/context_gpu.h"

namespace caffe2 {
REGISTER_CUDA_OPERATOR(Conv, ConvOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(ConvGradient, ConvGradientOp<float, CUDAContext>);

REGISTER_CUDA_OPERATOR(Conv1D, ConvOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(Conv1DGradient, ConvGradientOp<float, CUDAContext>);

REGISTER_CUDA_OPERATOR(Conv2D, ConvOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(Conv2DGradient, ConvGradientOp<float, CUDAContext>);

REGISTER_CUDA_OPERATOR(Conv3D, ConvOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(Conv3DGradient, ConvGradientOp<float, CUDAContext>);
}  // namespace caffe2
