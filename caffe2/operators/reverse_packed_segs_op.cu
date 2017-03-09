#include "caffe2/core/context_gpu.h"
#include "reverse_packed_segs_op.h"

namespace caffe2 {
namespace {
REGISTER_CUDA_OPERATOR(ReversePackedSegs, ReversePackedSegsOp<CUDAContext>);

} // namespace
} // namespace caffe2
