#pragma once
#include <string>

namespace torch {
namespace distributed {
namespace rpc {
// All metrics are prefixed with the following  key.
constexpr char kRPCMetricsKeyPrefix[] = "torch.distributed.rpc.";
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
  virtual ~RpcMetricsHandler() {}
};

} // namespace rpc
} // namespace distributed
} // namespace torch
