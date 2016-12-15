#include "redis_store_handler_op.h"

#include <caffe2/core/logging.h>

namespace caffe2 {

RedisStoreHandlerCreateOp::RedisStoreHandlerCreateOp(
    const OperatorDef& operator_def,
    Workspace* ws)
    : Operator(operator_def, ws),
      host_(GetSingleArgument<std::string>("host", "")),
      port_(GetSingleArgument<int>("port", 0)),
      prefix_(GetSingleArgument<std::string>("prefix", "")) {
  CAFFE_ENFORCE_NE(host_, "", "host is a required argument");
  CAFFE_ENFORCE_NE(port_, 0, "port is a required argument");
}

bool RedisStoreHandlerCreateOp::RunOnDevice() {
  auto ptr = std::unique_ptr<StoreHandler>(
      new RedisStoreHandler(host_, port_, prefix_));
  *OperatorBase::Output<std::unique_ptr<StoreHandler>>(HANDLER) =
      std::move(ptr);
  return true;
}

REGISTER_CPU_OPERATOR(RedisStoreHandlerCreate, RedisStoreHandlerCreateOp);
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
}
