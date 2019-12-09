#pragma once

#include "observers/macros.h"
#include "observers/net_observer_reporter.h"

#include "caffe2/core/common.h"

namespace caffe2 {

class CAFFE2_OBSERVER_API NetObserverReporterPrint : public NetObserverReporter {
 public:
  static const std::string IDENTIFIER;
  void report(NetBase* net, std::map<std::string, PerformanceInformation>&);
};

} // namespace caffe2
