#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/conv_transpose_op.h"
#include "caffe2/operators/conv_transpose_op_impl.h"

namespace caffe2 {
REGISTER_CUDA_OPERATOR(ConvTranspose, ConvTransposeOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    ConvTransposeGradient,
    ConvTransposeGradientOp<float, CUDAContext>);
} // namespace caffe2
