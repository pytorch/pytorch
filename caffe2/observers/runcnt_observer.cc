#include "runcnt_observer.h"

namespace caffe2 {

RunCountOperatorObserver::RunCountOperatorObserver(
    OperatorBase* op,
    RunCountNetObserver* netObserver)
    : ObserverBase<OperatorBase>(op), netObserver_(netObserver) {
  CAFFE_ENFORCE(netObserver_, "Observers can't operate outside of the net");
}

std::string RunCountNetObserver::debugInfo() {
#ifdef C10_ANDROID
  // workaround
  int foo = cnt_;
  return "This operator runs " + c10::to_string(foo) + " times.";
#else
  return "This operator runs " + c10::to_string(cnt_) + " times.";
#endif
}

void RunCountNetObserver::Start() {}

void RunCountNetObserver::Stop() {}

void RunCountOperatorObserver::Start() {
  ++netObserver_->cnt_;
}
void RunCountOperatorObserver::Stop() {}

std::unique_ptr<ObserverBase<OperatorBase>> RunCountOperatorObserver::rnnCopy(
    OperatorBase* subject,
    int rnn_order) const {
  return std::unique_ptr<ObserverBase<OperatorBase>>(
      new RunCountOperatorObserver(subject, netObserver_));
}

} // namespace caffe2
