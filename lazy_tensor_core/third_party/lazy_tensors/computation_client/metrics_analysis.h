#ifndef COMPUTATION_CLIENT_METRICS_ANALYSIS_H_
#define COMPUTATION_CLIENT_METRICS_ANALYSIS_H_

#include <iostream>
#include <memory>
#include <vector>

namespace lazy_tensors {
namespace metrics {

// Performance degradation symptoms detected:
// - Dynamic graphs
// - Very slow graph compilation
// - Very slow graph execution
// - Frequent device -> CPU transfers
// - Device HBM to host RAM swapping and HBM defragmentation
// - Unlowered aten:: ops

struct Analysis {
  enum class Symptom {
    kNormal,
    kMetricTooFrequent,
    kMetricTooSlow,
    kUnloweredOp,
  };

  Analysis() = default;
  Analysis(Symptom symptom) : symptom(symptom) {}
  Analysis(Symptom symptom, std::string repr) : symptom(symptom), repr(repr) {}

  Symptom symptom;
  std::string repr;
};

class Analyzer {
 public:
  virtual Analysis Run() = 0;
};

inline std::string CreatePerformanceReport() {
  LTC_LOG(FATAL) << "Not implemented.";
}

}  // namespace metrics
}  // namespace lazy_tensors

#endif  // COMPUTATION_CLIENT_METRICS_ANALYSIS_H_
