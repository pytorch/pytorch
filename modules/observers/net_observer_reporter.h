#pragma once

#include <map>

#include "caffe2/core/common.h"
#include "caffe2/core/net.h"
#include "observers/macros.h"

namespace caffe2 {

struct PerformanceInformation {
  // Analytic
  int64_t flops = 0;
  int64_t bytes_written = 0;
  int64_t bytes_read = 0;
  std::vector<TensorShape> tensor_shapes = {};
  std::vector<Argument> args = {};
  std::string engine = ""; // the engine used
  std::string type = ""; // the type of the operator
  // Measured
  double latency = 0;
  double cpuMilliseconds = 0;
};

class CAFFE2_OBSERVER_API NetObserverReporter {
 public:
  virtual ~NetObserverReporter() = default;

  /*
    Report the delay metric collected by the observer.
    The delays are saved in a map. The key is an identifier associated
    with the reported delay. The value is the delay value in float
  */
  virtual void report(
      NetBase* net,
      std::map<std::string, PerformanceInformation>&) = 0;
};
}
