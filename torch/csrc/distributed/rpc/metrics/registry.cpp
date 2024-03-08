#include <torch/csrc/distributed/rpc/metrics/RpcMetricsHandler.h> // @manual
namespace torch {
namespace distributed {
namespace rpc {
C10_DEFINE_REGISTRY(
    RpcMetricsHandlerRegistry,
    torch::distributed::rpc::RpcMetricsHandler);
} // namespace rpc
} // namespace distributed
} // namespace torch
