#include "caffe2/share/contrib/observers/net_observer_reporter_print.h"

#include "caffe2/core/init.h"
#include "caffe2/share/contrib/observers/observer_config.h"

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

void NetObserverReporterPrint::printNet(NetBase* net, double net_delay) {
  LOG(INFO) << IDENTIFIER << "Net Name - " << net->Name() << " :  Net Delay - "
            << net_delay;
}

void NetObserverReporterPrint::printNetWithOperators(
    NetBase* net,
    double net_delay,
    std::vector<std::pair<std::string, double>>& delays) {
  LOG(INFO) << IDENTIFIER << "Operators Delay Start";
  for (auto& p : delays) {
    LOG(INFO) << IDENTIFIER << p.first << " - " << p.second;
  }
  LOG(INFO) << IDENTIFIER << "Operators Delay End";
}
}
