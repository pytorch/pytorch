#include "observers/net_observer_reporter_print.h"

#include <algorithm>
#include <sstream>
#include "caffe2/core/init.h"
#include "observers/observer_config.h"

#include <c10/util/irange.h>

namespace caffe2 {

const std::string NetObserverReporterPrint::IDENTIFIER = "Caffe2Observer ";
static std::string get_op_args(PerformanceInformation p);
static std::string get_tensor_shapes(PerformanceInformation p);
static std::string sanatize(std::string json_s);

void NetObserverReporterPrint::report(
    NetBase* net,
    std::map<std::string, PerformanceInformation>& info) {
  // Not allowed to use json library
  std::vector<std::map<std::string, std::string>> caffe2_perf;

  for (auto& p : info) {
    if ((p.first == "NET_DELAY") && (info.size() == 1)) {
      // for Net_delay perf
      caffe2_perf.push_back({{"type", "NET"},
                             {"value", c10::to_string(p.second.latency * 1000)},
                             {"unit", "us"},
                             {"metric", "latency"}});
      caffe2_perf.push_back({{"type", "NET_"},
                             {
                               "value",
                               c10::to_string(
                                   p.second.cpuMilliseconds /
                                   p.second.latency *
                                   100),
                             },
                             {"unit", "percent"},
                             {"metric", "cpu_percent"}});
    } else if (p.first != "NET_DELAY") {
      // for operator perf
      std::string shape_str = get_tensor_shapes(p.second);
      std::string args_str = get_op_args(p.second);
      std::string type = p.first;
      caffe2_perf.push_back({{"type", type},
                             {"value", c10::to_string(p.second.latency * 1000)},
                             {"unit", "us"},
                             {"metric", "latency"}});
      caffe2_perf.push_back({{"type", type},
                             {
                               "value",
                               c10::to_string(
                                   p.second.cpuMilliseconds /
                                   p.second.latency *
                                   100),
                             },
                             {"unit", "percent"},
                             {"metric", "cpu_percent"}});
      if (p.second.flops > 0) {
        caffe2_perf.push_back({{"type", type},
                               {"value", c10::to_string(p.second.flops)},
                               {"unit", "flop"},
                               {"metric", "flops"}});
      }
      if (shape_str != "") {
        caffe2_perf.push_back({{"type", type},
                               {"info_string", shape_str},
                               {"unit", ""},
                               {"metric", "tensor_shapes"}});
      }
      if (args_str != "") {
        caffe2_perf.push_back({{"type", type},
                               {"info_string", args_str},
                               {"unit", ""},
                               {"metric", "op_args"}});
      }
    }
  }

  // NOLINTNEXTLINE(modernize-loop-convert)
  for (auto it = caffe2_perf.begin(); it != caffe2_perf.end(); it++) {
    std::stringstream buffer;
    auto entry = *it;
    buffer << IDENTIFIER << "{";
    // NOLINTNEXTLINE(modernize-raw-string-literal)
    buffer << "\"type\": \"" << sanatize(entry["type"]) << "\","
           // NOLINTNEXTLINE(modernize-raw-string-literal)
           << "\"unit\": \"" << sanatize(entry["unit"]) << "\","
           // NOLINTNEXTLINE(modernize-raw-string-literal)
           << "\"metric\": \"" << sanatize(entry["metric"]) << "\",";
    if (entry.find("value") != entry.end()) {
      // NOLINTNEXTLINE(modernize-raw-string-literal)
      buffer << "\"value\": \"" << sanatize(entry["value"]) << "\"";
    } else if (entry.find("info_string") != entry.end()) {
      // NOLINTNEXTLINE(modernize-raw-string-literal)
      buffer << "\"info_string\": \"" << sanatize(entry["info_string"]) << "\"";
    }
    buffer << "}";
    LOG(INFO) << buffer.str();
  }
}

static std::string get_tensor_shapes(PerformanceInformation p) {
  std::string shape_str;
  std::stringstream shape_stream;
  if (!p.tensor_shapes.empty()) {
    shape_stream << "[";
    for (const auto i : c10::irange(p.tensor_shapes.size())) {
      shape_stream << "[";
      for (int j = 0; j < p.tensor_shapes[i].dims_size(); j++) {
        shape_stream << p.tensor_shapes[i].dims(j) << ", ";
      }
      shape_stream << "], ";
    }
    shape_stream << "]";
    shape_str = shape_stream.str();
  } else {
    shape_str = "";
  }
  return shape_str;
}

static std::string get_op_args(PerformanceInformation p) {
  std::string args_str;
  if (!p.args.empty()) {
    std::stringstream args;
    args << "[";
    for (const auto i : c10::irange(p.args.size())) {
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
    args_str = "";
  }
  return args_str;
}

static std::string sanatize(std::string json_s) {
  // Remove illegal characters from the name that would cause json string to
  // become invalid
  json_s.erase(std::remove(json_s.begin(), json_s.end(), '"'), json_s.end());
  json_s.erase(std::remove(json_s.begin(), json_s.end(), '\\'), json_s.end());
  return json_s;
}
}
