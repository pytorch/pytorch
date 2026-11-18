#include <torch/csrc/distributed/rpc/metrics/RpcMetricsHandler.h> // @manual

namespace torch::distributed::rpc {
C10_DEFINE_REGISTRY(
    RpcMetricsHandlerRegistry,
    torch::distributed::rpc::RpcMetricsHandler)
} // namespace torch::distributed::rpc
