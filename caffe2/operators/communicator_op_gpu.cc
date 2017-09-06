#include "caffe2/core/context_gpu.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/no_default_engine_op.h"

namespace caffe2 {
// Communication operators do not have default engines.
REGISTER_CUDA_OPERATOR(CreateCommonWorld, NoDefaultEngineOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(CloneCommonWorld, NoDefaultEngineOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(Broadcast, NoDefaultEngineOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(Reduce, NoDefaultEngineOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(Allgather, NoDefaultEngineOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(Allreduce, NoDefaultEngineOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(SendTensor, NoDefaultEngineOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(ReceiveTensor, NoDefaultEngineOp<CUDAContext>);

} // namespace caffe2
