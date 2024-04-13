#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/counter_ops.h"

namespace caffe2 {
REGISTER_CUDA_OPERATOR(CreateCounter, CreateCounterOp<int64_t, CUDAContext>);
REGISTER_CUDA_OPERATOR(ResetCounter, ResetCounterOp<int64_t, CUDAContext>);
REGISTER_CUDA_OPERATOR(CountDown, CountDownOp<int64_t, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    CheckCounterDone,
    CheckCounterDoneOp<int64_t, CUDAContext>);
REGISTER_CUDA_OPERATOR(CountUp, CountUpOp<int64_t, CUDAContext>);
REGISTER_CUDA_OPERATOR(RetrieveCount, RetrieveCountOp<int64_t, CUDAContext>);
} // namespace caffe2
