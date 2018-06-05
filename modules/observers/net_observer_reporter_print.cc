#include "observers/net_observer_reporter_print.h"

#include "caffe2/core/init.h"
#include "observers/observer_config.h"

namespace caffe2 {

const std::string NetObserverReporterPrint::IDENTIFIER = "Caffe2Observer ";

void NetObserverReporterPrint::report(
    NetBase* net,
    std::map<std::string, PerformanceInformation>& info) {
  LOG(INFO) << IDENTIFIER << "Net Name - " << net->Name();
  LOG(INFO) << IDENTIFIER << "Delay Start";
  for (auto& p : info) {
    LOG(INFO) << IDENTIFIER << p.first << " - " << p.second.latency << "\t(ms)";
  }
  LOG(INFO) << IDENTIFIER << "Delay End";
}
}
