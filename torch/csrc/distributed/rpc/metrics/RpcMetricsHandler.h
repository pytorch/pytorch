#pragma once
#include <c10/util/Registry.h>
#include <string>

namespace torch {
namespace distributed {
namespace rpc {
// All metrics are prefixed with the following  key.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
constexpr char kRpcMetricsKeyPrefix[] = "torch.distributed.rpc.";
// APIs for logging time-series metrics for RPC-based distributed
// training. Implementations of this class should provide thread safety so that
// metrics can be logged from multiple threads without the user needing to
// coordinate serialization.
class RpcMetricsHandler {
 public:
  // Accumulates the metric value specified by the name for purposes of
  // computing aggregate statistics over time.
  virtual void accumulateMetric(const std::string& name, double value) = 0;
  // Increment a count for the metric given by the name.
  virtual void incrementMetric(const std::string& name) = 0;
  // NOLINTNEXTLINE(modernize-use-equals-default)
  virtual ~RpcMetricsHandler() {}
};

// Configuration struct for metrics handling.
struct RpcMetricsConfig {
  explicit RpcMetricsConfig(std::string handlerName, bool enabled)
      : handlerName_(std::move(handlerName)), enabled_(enabled) {}

  // Handler name
  std::string handlerName_;
  // Whether metrics exporting should be enabled or not.
  bool enabled_;
};

// A registry for different implementations of RpcMetricsHandler. Classes
// implementing the above interface should use this to register implementations.
C10_DECLARE_REGISTRY(
    RpcMetricsHandlerRegistry,
    torch::distributed::rpc::RpcMetricsHandler);

} // namespace rpc
} // namespace distributed
} // namespace torch
