#include "redis_store_handler_op.h"

#include <caffe2/core/context_gpu.h>

namespace caffe2 {

REGISTER_CPU_OPERATOR(
    RedisStoreHandlerCreate,
    RedisStoreHandlerCreateOp<CPUContext>);

REGISTER_CUDA_OPERATOR(
    RedisStoreHandlerCreate,
    RedisStoreHandlerCreateOp<CUDAContext>);

OPERATOR_SCHEMA(RedisStoreHandlerCreate)
    .NumInputs(0)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Creates a unique_ptr<StoreHandler> that uses a Redis server as backing store.
)DOC")
    .Arg("host", "host name of Redis server")
    .Arg("port", "port number of Redis server")
    .Arg("prefix", "keys used by this instance are prefixed with this string")
    .Output(0, "handler", "unique_ptr<StoreHandler>");

NO_GRADIENT(RedisStoreHandlerCreateOp);

} // namespace caffe2
