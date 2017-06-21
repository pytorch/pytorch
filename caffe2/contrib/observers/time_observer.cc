#include "caffe2/contrib/observers/time_observer.h"
#include "caffe2/core/logging.h"

namespace caffe2 {

template <>
bool TimeObserver<NetBase>::Start() {
  vector<OperatorBase*> operators = subject_->getOperators();
  for (auto& op : operators) {
    children_.push_back(caffe2::make_unique<TimeObserver<OperatorBase>>(op));
  }
  start_time_ = timer_.MilliSeconds();
  ++iterations_;
  return true;
}

template <>
bool TimeObserver<NetBase>::Stop() {
  double current_run = timer_.MilliSeconds() - start_time_;
  total_time_ += current_run;
  VLOG(1) << "This net iteration took " << current_run << " ms to complete.\n";
  return true;
}

template <>
bool TimeObserver<OperatorBase>::Start() {
  start_time_ = timer_.MilliSeconds();
  ++iterations_;
  return true;
}

template <>
bool TimeObserver<OperatorBase>::Stop() {
  double current_run = timer_.MilliSeconds() - start_time_;
  total_time_ += current_run;
  VLOG(1) << "This operator iteration took " << current_run
          << " ms to complete.\n";
  return true;
}

} // namespace caffe2
