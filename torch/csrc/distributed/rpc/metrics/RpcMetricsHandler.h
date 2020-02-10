#pragma once
#include <string>

namespace torch {
namespace distributed {
namespace rpc {
// All metrics are prefixed with the following  key.
const std::string kRPCMetricsKeyPrefix = "torch.distributed.rpc.";
// Thread-safe API for logging time-series metrics for RPC-based model parallel
// training. This class provides thread safety so that metrics can be logged
// from multiple threads without the user needing to coordinate serialization.
// The class provides two main functions - accumulateMetric which can be used
// for statistical aggregations and setMetric for setting/modifying single
// values.
class RpcMetricsHandler {
 public:
  // Log value corresponding to metric name given by name to the time-series
  // metric stream.
  virtual void accumulateMetric(const std::string& name, double value) = 0;
  // Increment a count for the metric given by the name.
  virtual void bumpMetric(const std::string& name) = 0;
  virtual ~RpcMetricsHandler() {}
};

} // namespace rpc
} // namespace distributed
} // namespace torch
