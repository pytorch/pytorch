#include "observers/net_observer_reporter_print.h"

#include <sstream>
#include "caffe2/core/init.h"
#include "observers/observer_config.h"

namespace caffe2 {

const std::string NetObserverReporterPrint::IDENTIFIER = "Caffe2Observer ";

void NetObserverReporterPrint::report(
    NetBase* net,
    std::map<std::string, PerformanceInformation>& info) {
  std::map<std::string, std::map<std::string, std::string>> caffe2_perf;
  std::map<std::string, std::string> op_perf;
  std::map<std::string, std::string> net_perf;

  for (auto& p : info) {
    if ((p.first == "NET_DELAY") && (info.size() == 1)) {
      // for Net_delay perf
      net_perf["latency"] = caffe2::to_string(p.second.latency);
      net_perf["flops"] = caffe2::to_string(-1);
      caffe2_perf["NET"] = net_perf;
    } else if (p.first != "NET_DELAY") {
      // for operator perf
      op_perf["latency"] = caffe2::to_string(p.second.latency);
      op_perf["flops"] = caffe2::to_string(p.second.flops);
      caffe2_perf[p.first] = op_perf;
    }
  }

  std::stringstream buffer;
  buffer << IDENTIFIER << "{";
  for (auto it = caffe2_perf.begin(); it != caffe2_perf.end(); it++) {
    buffer << "\"" << it->first << "\""
           << ": {";
    for (auto jt = it->second.begin(); jt != it->second.end(); jt++) {
      buffer << "\"" << jt->first << "\""
             << ": " << jt->second;
      auto kt = jt;
      if ((++kt) != it->second.end()) {
        buffer << ", ";
      }
    }
    buffer << "}";
    auto lt = it;
    if ((++lt) != caffe2_perf.end()) {
      buffer << ", ";
    }
  }
  buffer << "}";
  LOG(INFO) << buffer.str();
}
}
