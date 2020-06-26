// Copyright 2004-present Facebook. All Rights Reserved.
#include "RpcMetricsHandler.h"
namespace torch {
namespace distributed {
namespace rpc {
C10_DEFINE_REGISTRY(
    RpcMetricsHandlerRegistry,
    torch::distributed::rpc::RpcMetricsHandler);
}
} // namespace distributed
} // namespace torch
