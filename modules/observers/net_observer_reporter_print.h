#pragma once

#include "observers/net_observer_reporter.h"

namespace caffe2 {

class NetObserverReporterPrint : public NetObserverReporter {
 public:
  static const std::string IDENTIFIER;
  void reportDelay(
      NetBase* net,
      std::map<std::string, double>& delays,
      const char* unit);
};
}
