#include "runcnt_observer.h"

namespace caffe2 {

RunCountOperatorObserver::RunCountOperatorObserver(
    OperatorBase* op,
    RunCountNetObserver* netObserver)
    : ObserverBase<OperatorBase>(op), netObserver_(netObserver) {
  CAFFE_ENFORCE(netObserver_, "Observers can't operate outside of the net");
}

std::unique_ptr<ObserverBase<OperatorBase>> RunCountOperatorObserver::clone() {
  return std::unique_ptr<ObserverBase<OperatorBase>>(
      new RunCountOperatorObserver(this->subject_, netObserver_));
}

std::string RunCountNetObserver::debugInfo() {
  return "This operator runs " + caffe2::to_string(cnt_) + " times.";
}

bool RunCountNetObserver::Start() {
  const auto& operators = subject_->GetOperators();
  for (auto* op : operators) {
    op->AttachObserver(caffe2::make_unique<RunCountOperatorObserver>(op, this));
  }
  return true;
}

bool RunCountNetObserver::Stop() {
  return true;
}

bool RunCountOperatorObserver::Start() {
  ++netObserver_->cnt_;
  return true;
}
bool RunCountOperatorObserver::Stop() {
  return true;
}

} // namespace caffe2
