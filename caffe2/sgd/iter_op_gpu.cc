#include "caffe2/core/context_gpu.h"
#include "caffe2/sgd/iter_op.h"

namespace caffe2 {
namespace {
REGISTER_CUDA_OPERATOR(Iter, IterOp<CUDAContext>);
}
} // namespace caffe2
