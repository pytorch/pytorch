#pragma once

#include "caffe2/core/net.h"
#include "caffe2/core/observer.h"
#include "caffe2/core/timer.h"

#include <unordered_map>

namespace caffe2 {

class PerfNetObserver : public NetObserver {
 public:
  explicit PerfNetObserver(NetBase* subject_);
  virtual ~PerfNetObserver();

  caffe2::Timer& getTimer() {
    return timer_;
  }

 private:
  void Start() override;
  void Stop() override;

  caffe2::string getObserverName(const OperatorBase* op, int idx) const;

 private:
  enum LogType {
    NONE,
    OPERATOR_DELAY,
    NET_DELAY,
  };
  LogType logType_;
  unsigned int numRuns_;
  std::unordered_map<const OperatorBase*, const ObserverBase<OperatorBase>*>
      observerMap_;

  caffe2::Timer timer_;
};

class PerfOperatorObserver : public ObserverBase<OperatorBase> {
 public:
  PerfOperatorObserver(OperatorBase* op, PerfNetObserver* netObserver);
  virtual ~PerfOperatorObserver();

  double getMilliseconds() const;

 private:
  void Start() override;
  void Stop() override;

 private:
  // Observer of a net that owns corresponding op. We make sure net is never
  // destructed while operator observer is still alive. First operator observer
  // gets destructed, then the op, then the net and its observer.
  // We do this trick in order to get access to net's name and other fields
  // without storing inside the operator observer. Each field is memory
  // costly here and a raw pointer is a cheapest sholution
  PerfNetObserver* netObserver_;
  double milliseconds_;
};
} // namespace caffe2
