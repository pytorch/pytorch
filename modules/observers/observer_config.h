#pragma once

#include "observers/net_observer_reporter.h"

namespace caffe2 {

/*
  netInitSampleRate_ == 1 && operatorNetSampleRatio_ == 1 :
      Log operator metrics in every iteration
  netInitSampleRate_ == 1 && operatorNetSampleRatio_ == 0 :
      Log net metrics in every iterationn
  netInitSampleRate_ == n && netFollowupSampleRate_ == m &&
          netFollowupSampleCount == c && operatorNetSampleRatio_ == 1 :
      Log operator metrics first at odds of 1 / n. Once first logged,
      the following c logs are at odds of 1 / min(n, m). Then repeat
  netInitSampleRate_ == n && netFollowupSampleRate_ == m &&
          netFollowupSampleCount == c && operatorNetSampleRatio_ == 0 :
      Log net metrics first at odds of 1 / n. Once first logged,
      the following c logs are at odds of 1 / min(n, m). Then repeat
  netInitSampleRate_ == n && netFollowupSampleRate_ == m &&
          netFollowupSampleCount == c && operatorNetSampleRatio_ == o :
      Log net metrics first at odds of 1 / n. Once first logged,
      the following c logs are at odds of 1 / min(n, m), if the random number
      is multiples of o, log operator metrics instead. Then repeat
  skipIters_ == n: skip the first n iterations of the net.
*/
class ObserverConfig {
 public:
  static void initSampleRate(
      int netInitSampleRate,
      int netFollowupSampleRate,
      int netFollowupSampleCount,
      int operatorNetSampleRatio,
      int skipIters) {
    CAFFE_ENFORCE(netFollowupSampleRate <= netInitSampleRate);
    CAFFE_ENFORCE(netFollowupSampleRate >= 1 || netInitSampleRate == 0);
    netInitSampleRate_ = netInitSampleRate;
    netFollowupSampleRate_ = netFollowupSampleRate;
    netFollowupSampleCount_ = netFollowupSampleCount;
    operatorNetSampleRatio_ = operatorNetSampleRatio;
    skipIters_ = skipIters;
  }
  static int getNetInitSampleRate() {
    return netInitSampleRate_;
  }
  static int getNetFollowupSampleRate() {
    return netFollowupSampleRate_;
  }
  static int getNetFollowupSampleCount() {
    return netFollowupSampleCount_;
  }
  static int getOpoeratorNetSampleRatio() {
    return operatorNetSampleRatio_;
  }
  static int getSkipIters() {
    return skipIters_;
  }
  static void setReporter(unique_ptr<NetObserverReporter> reporter) {
    // Can only set the reporter once
    CAFFE_ENFORCE(reporter_ == nullptr);
    reporter_ = std::move(reporter);
  }
  static NetObserverReporter* getReporter() {
    CAFFE_ENFORCE(reporter_);
    return reporter_.get();
  }
  static void setMarker(int marker) {
    marker_ = marker;
  }
  static int getMarker() {
    return marker_;
  }

 private:
  /* The odds of log net metric initially or immediately after reset */
  static int netInitSampleRate_;

  /* The odds of log net metric after log once after start of reset */
  static int netFollowupSampleRate_;

  /* The number of follow up logs to be collected for odds of
     netFollowupSampleRate_ */
  static int netFollowupSampleCount_;

  /* The odds to log the operator metric instead of the net metric.
     When the operator is logged the net is not logged. */
  static int operatorNetSampleRatio_;

  /* skip the first few iterations */
  static int skipIters_;

  static unique_ptr<NetObserverReporter> reporter_;

  /* marker used in identifying the metrics in certain reporters */
  static int marker_;
};

}
