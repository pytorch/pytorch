#pragma once

#include "caffe2/share/contrib/observers/net_observer_reporter.h"

namespace caffe2 {

class NetObserverReporterPrint : public NetObserverReporter {
 public:
  static const std::string IDENTIFIER;
  void printNet(NetBase* net, double net_delay);
  void printNetWithOperators(
      NetBase* net,
      double net_delay,
      std::vector<std::pair<std::string, double>>& operator_delays);
};
}
