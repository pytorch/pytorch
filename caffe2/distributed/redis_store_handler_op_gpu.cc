#include "redis_store_handler_op.h"

#include <caffe2/core/context_gpu.h>

namespace caffe2 {

REGISTER_CUDA_OPERATOR(
    RedisStoreHandlerCreate,
    RedisStoreHandlerCreateOp<CUDAContext>);

} // namespace caffe2
