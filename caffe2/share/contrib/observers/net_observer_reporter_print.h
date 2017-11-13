#pragma once

#include "caffe2/share/contrib/observers/net_observer_reporter.h"

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
