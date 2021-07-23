#include <torch/csrc/distributed/rpc/metrics/RpcMetricsHandler.h> // @manual
namespace torch {
namespace distributed {
namespace rpc {
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
C10_DEFINE_REGISTRY(
    RpcMetricsHandlerRegistry,
    torch::distributed::rpc::RpcMetricsHandler);
} // namespace rpc
} // namespace distributed
} // namespace torch
