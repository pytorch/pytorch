#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/utility_ops.h"

namespace caffe2 {
namespace {

REGISTER_CUDA_OPERATOR(Print, PrintOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(Flatten, FlattenOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(Alias, AliasOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(ResizeLike, ResizeLikeOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(Sum, SumOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(WeightedSum, WeightedSumOp<float, CUDAContext>);
// CopyGPUToCPU and CopyCPUToGPU should both be carried out in a cuda context,
// since gpu code will be involved.
REGISTER_CUDA_OPERATOR(CopyGPUToCPU,
                       CopyOp<CUDAContext, CPUContext, CUDAContext>);
REGISTER_CUDA_OPERATOR(CopyCPUToGPU,
                       CopyOp<CUDAContext, CUDAContext, CPUContext>);
// If we only specify Copy, we assume that it is a gpu to gpu copy - maybe
// involving different GPUs.
REGISTER_CUDA_OPERATOR(Copy,
                       CopyOp<CUDAContext, CUDAContext, CUDAContext>);

}  // namespace
}  // namespace caffe2


