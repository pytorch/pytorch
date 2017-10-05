#pragma once
#include <utility>
#include <vector>
#include "caffe2/core/net.h"

namespace caffe2 {

class NetObserverReporter {
 public:
  virtual void printNet(NetBase* net, double net_delay) = 0;
  virtual void printNetWithOperators(
      NetBase* net,
      double net_delay,
      std::vector<std::pair<std::string, double>>& operator_delays) = 0;
};
}
