#pragma once
#include <map>
#include "caffe2/core/net.h"

namespace caffe2 {

class NetObserverReporter {
 public:
  virtual ~NetObserverReporter() = default;

  /*
    Report the delay metric collected by the observer.
    The delays are saved in a map. The key is an identifier associated
    with the reported delay. The value is the delay value in float
  */
  virtual void reportDelay(
      NetBase* net,
      std::map<std::string, double>& delays,
      const char* unit) = 0;
};
}
