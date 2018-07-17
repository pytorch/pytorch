#include "observers/net_observer_reporter_print.h"

#include <sstream>
#include "caffe2/core/init.h"
#include "observers/observer_config.h"

namespace caffe2 {

const std::string NetObserverReporterPrint::IDENTIFIER = "Caffe2Observer ";
static std::string get_op_args(PerformanceInformation p);
static std::string get_tensor_shapes(PerformanceInformation p);

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
      std::string shape_str = get_tensor_shapes(p.second);
      std::string args_str = get_op_args(p.second);

      caffe2_perf[p.first] = {
          {"latency",
           {{"value", caffe2::to_string(p.second.latency * 1000)},
            {"unit", "us"}}},
          {"flops",
           {{
                "value",
                caffe2::to_string(p.second.flops),
            },
            {"unit", "flops"}}},
          {"tensor_shapes", {{"info_string", shape_str}, {"unit", ""}}},
          {"op_args", {{"info_string", args_str}, {"unit", ""}}}};
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

static std::string get_tensor_shapes(PerformanceInformation p) {
  std::string shape_str;
  std::stringstream shape_stream;
  if (!p.tensor_shapes.empty()) {
    shape_stream << "[";
    for (int i = 0; i < p.tensor_shapes.size(); i++) {
      shape_stream << "[";
      for (int j = 0; j < p.tensor_shapes[i].dims_size(); j++) {
        shape_stream << p.tensor_shapes[i].dims(j) << ", ";
      }
      shape_stream << "], ";
    }
    shape_stream << "]";
    shape_str = shape_stream.str();
  } else {
    shape_str = "[]";
  }
  return shape_str;
}

static std::string get_op_args(PerformanceInformation p) {
  std::string args_str;
  if (!p.args.empty()) {
    std::stringstream args;
    args << "[";
    for (int i = 0; i < p.args.size(); i++) {
      args << "{" << p.args[i].name() << ": ";
      if (p.args[i].has_i()) {
        args << p.args[i].i();
      } else if (p.args[i].has_s()) {
        args << p.args[i].s();
      } else if (p.args[i].has_n()) {
        args << &p.args[i].n();
      } else if (p.args[i].has_f()) {
        args << p.args[i].f();
      } else {
        args << "None";
      }
      args << "}, ";
    }
    args << "]";
    args_str = args.str();
  } else {
    args_str = "[]";
  }
  return args_str;
}
}
