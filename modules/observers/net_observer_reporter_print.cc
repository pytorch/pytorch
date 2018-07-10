#include "observers/net_observer_reporter_print.h"

#include <sstream>
#include "caffe2/core/init.h"
#include "observers/observer_config.h"

namespace caffe2 {

const std::string NetObserverReporterPrint::IDENTIFIER = "Caffe2Observer ";

void NetObserverReporterPrint::report(
    NetBase* net,
    std::map<std::string, PerformanceInformation>& info) {
  // Not allowed to use json library
  std::map<
      std::string,
      std::map<std::string, std::map<std::string, std::string>>>
      caffe2_perf;

  for (auto& p : info) {
    if ((p.first == "NET_DELAY") && (info.size() == 1)) {
      // for Net_delay perf
      caffe2_perf["NET"] = {
          {"latency",
           {{"value", caffe2::to_string(p.second.latency * 1000)},
            {"unit", "us"}}},
          {"flops", {{"value", "-1"}, {"unit", "flops"}}}};
    } else if (p.first != "NET_DELAY") {
      // for operator perf
      caffe2_perf[p.first] = {
          {"latency",
           {{"value", caffe2::to_string(p.second.latency * 1000)},
            {"unit", "us"}}},
          {"flops",
           {{
                "value",
                caffe2::to_string(p.second.flops),
            },
            {"unit", "flops"}}}};
    }
  }

  for (auto it = caffe2_perf.begin(); it != caffe2_perf.end(); it++) {
    std::stringstream buffer;
    buffer << IDENTIFIER << "{";
    buffer << "\"" << it->first << "\""
           << ": {";
    for (auto jt = it->second.begin(); jt != it->second.end(); jt++) {
      buffer << "\"" << jt->first << "\""
             << ": {";
      for (auto kt = jt->second.begin(); kt != jt->second.end(); kt++) {
        buffer << "\"" << kt->first << "\""
               << ": "
               << "\"" << kt->second << "\"";
        auto lt = kt;
        if ((++lt) != jt->second.end()) {
          buffer << ", ";
        }
      }
      buffer << "}";
      auto lt = jt;
      if ((++lt) != it->second.end()) {
        buffer << ", ";
      }
    }
    buffer << "}}";
    LOG(INFO) << buffer.str();
  }
}
}
