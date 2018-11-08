#include "time_observer.h"
#include "caffe2/core/logging.h"

namespace caffe2 {

void TimeObserver::Start() {
  start_time_ = timer_.MilliSeconds();
  ++iterations_;
}

void TimeObserver::Stop() {
  double current_run = timer_.MilliSeconds() - start_time_;
  total_time_ += current_run;
  VLOG(1) << "This net iteration took " << current_run << " ms to complete.\n";
}

void TimeOperatorObserver::Start() {
  start_time_ = timer_.MilliSeconds();
  ++iterations_;
}

void TimeOperatorObserver::Stop() {
  double current_run = timer_.MilliSeconds() - start_time_;
  total_time_ += current_run;
  VLOG(1) << "This operator iteration took " << current_run
          << " ms to complete.\n";
}

std::unique_ptr<ObserverBase<OperatorBase>> TimeOperatorObserver::rnnCopy(
    OperatorBase* subject,
    int rnn_order) const {
  return std::unique_ptr<ObserverBase<OperatorBase>>(
      new TimeOperatorObserver(subject, nullptr));
}

} // namespace caffe2
