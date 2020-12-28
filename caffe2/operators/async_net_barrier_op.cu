#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/async_net_barrier_op.h"

namespace caffe2 {

REGISTER_CUDA_OPERATOR(AsyncNetBarrier, AsyncNetBarrierOp<CUDAContext>);

} // namespace caffe2
