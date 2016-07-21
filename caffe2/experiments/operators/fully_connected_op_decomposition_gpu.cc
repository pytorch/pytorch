#include "caffe2/core/context_gpu.h"
#include "caffe2/experiments/operators/fully_connected_op_decomposition.h"

namespace caffe2 {
namespace {
REGISTER_CUDA_OPERATOR(FC_Dcomp, FullyConnectedOpDecomp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(FCGradient_Decomp,
                       FullyConnectedDecompGradientOp<float, CUDAContext>);
}  // namespace
}  // namespace caffe2
