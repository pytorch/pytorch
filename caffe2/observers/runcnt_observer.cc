#include "runcnt_observer.h"

namespace caffe2 {

RunCountOperatorObserver::RunCountOperatorObserver(
    OperatorBase* op,
    RunCountNetObserver* netObserver)
    : ObserverBase<OperatorBase>(op), netObserver_(netObserver) {
  CAFFE_ENFORCE(netObserver_, "Observers can't operate outside of the net");
}

std::unique_ptr<ObserverBase<OperatorBase>> RunCountOperatorObserver::copy(
    OperatorBase* subject) {
  return std::unique_ptr<ObserverBase<OperatorBase>>(
      new RunCountOperatorObserver(subject, netObserver_));
}

std::string RunCountNetObserver::debugInfo() {
  return "This operator runs " + caffe2::to_string(cnt_) + " times.";
}

void RunCountNetObserver::Start() {
  const auto& operators = subject_->GetOperators();
  for (auto* op : operators) {
    op->AttachObserver(caffe2::make_unique<RunCountOperatorObserver>(op, this));
  }
}

void RunCountNetObserver::Stop() {}

void RunCountOperatorObserver::Start() {
  ++netObserver_->cnt_;
}
void RunCountOperatorObserver::Stop() {}

} // namespace caffe2
