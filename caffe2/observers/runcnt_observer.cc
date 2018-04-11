#include "runcnt_observer.h"

namespace caffe2 {

RunCountOperatorObserver::RunCountOperatorObserver(
    OperatorBase* op,
    RunCountNetObserver* netObserver)
    : RNNCapableOperatorObserver(op), netObserver_(netObserver) {
  CAFFE_ENFORCE(netObserver_, "Observers can't operate outside of the net");
}

std::string RunCountNetObserver::debugInfo() {
#if CAFFE2_ANDROID
  // workaround
  int foo = cnt_;
  return "This operator runs " + caffe2::to_string(foo) + " times.";
#else
  return "This operator runs " + caffe2::to_string(cnt_) + " times.";
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
