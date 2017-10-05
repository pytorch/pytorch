#pragma once

#include "caffe2/share/contrib/observers/net_observer_reporter.h"

namespace caffe2 {

/*
  netSampleRate_ == 1 && operatorNetSampleRatio_ == 1 :
      Log operator metrics in every iteration
  netSampleRate_ == 1 && operatorNetSampleRatio_ == 0 :
      Log net metrics in every iterationn
  netSampleRate_ == n && operatorNetSampleRatio_ == 1 :
      Log operator metrics every n iterations
  netSampleRate_ == n && operatorNetSampleRatio_ == 0 :
      Log net metrics every n iterations
  netSampleRate_ == n && operatorNetSampleRatio_ == m :
      Log net metrics every n iterations (except n * m iterations)
      Log operator metrics every n * m iterations
  skipIters_ == n: skip the first n iterations of the net.
*/
class ObserverConfig {
 public:
  static void
  initSampleRate(int netSampleRate, int operatorNetSampleRatio, int skipIters) {
    netSampleRate_ = netSampleRate;
    operatorNetSampleRatio_ = operatorNetSampleRatio;
    skipIters_ = skipIters;
  }
  static int getNetSampleRate() {
    return netSampleRate_;
  }
  static int getOpoeratorNetSampleRatio() {
    return operatorNetSampleRatio_;
  }
  static int getSkipIters() {
    return skipIters_;
  }
  static void setReporter(unique_ptr<NetObserverReporter> reporter) {
    reporter_ = std::move(reporter);
  }
  static NetObserverReporter* getReporter() {
    CAFFE_ENFORCE(reporter_);
    return reporter_.get();
  }

 private:
  /* Log net metrics after how many net invocations */
  static int netSampleRate_;

  /* log operator metrics after how many net logs.
     when the operator is logged the net is not logged. */
  static int operatorNetSampleRatio_;

  /* skip the first few iterations */
  static int skipIters_;

  static unique_ptr<NetObserverReporter> reporter_;
};

}
