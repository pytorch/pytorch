#include "observers/net_observer_reporter_print.h"

#include "caffe2/core/init.h"
#include "observers/observer_config.h"

namespace caffe2 {

namespace {
bool registerGlobalPerfNetObserverReporter(int* /*pargc*/, char*** /*pargv*/) {
  ObserverConfig::setReporter(make_unique<NetObserverReporterPrint>());
  return true;
}
} // namespace

REGISTER_CAFFE2_EARLY_INIT_FUNCTION(
    registerGlobalPerfNetObserverReporter,
    &registerGlobalPerfNetObserverReporter,
    "Caffe2 print net observer reporter");

const std::string NetObserverReporterPrint::IDENTIFIER = "Caffe2Observer ";

void NetObserverReporterPrint::reportDelay(
    NetBase* net,
    std::map<std::string, double>& delays,
    const char* unit) {
  CAFFE_ENFORCE(unit != nullptr, "Unit is null");
  LOG(INFO) << IDENTIFIER << "Net Name - " << net->Name();
  LOG(INFO) << IDENTIFIER << "Delay Start";
  for (auto& p : delays) {
    LOG(INFO) << IDENTIFIER << p.first << " - " << p.second << "\t(" << *unit
              << ")";
  }
  LOG(INFO) << IDENTIFIER << "Delay End";
}
}
