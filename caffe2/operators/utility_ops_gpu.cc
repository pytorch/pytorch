#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/utility_ops.h"

namespace caffe2 {
namespace {

REGISTER_CUDA_OPERATOR(Free, FreeOp);
REGISTER_CUDA_OPERATOR(Print, PrintOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(PrintInt, PrintOp<int, CUDAContext>);
REGISTER_CUDA_OPERATOR(Flatten, FlattenOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(Alias, FlattenOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(ReshapeLike, ReshapeLikeOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(Split, SplitOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(Sum, SumOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(WeightedSum, WeightedSumOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(CopyGPUToCPU,
                       CopyOp<float, CUDAContext, CPUContext, CUDAContext>);
REGISTER_CUDA_OPERATOR(CopyCPUToGPU,
                       CopyOp<float, CUDAContext, CUDAContext, CPUContext>);

}  // namespace
}  // namespace caffe2


