#include <htrace.hpp>
#include <algorithm>
#include <ctime>

#include "caffe2/core/net.h"
#include "caffe2/core/operator.h"

CAFFE2_DEFINE_string(caffe2_htrace_conf, "", "Configuration string for htrace");

namespace caffe2 {
namespace {

const string defaultHTraceConf(const string& net_name) {
  // create a duplicate because we may need to modify the name
  string net_name_copy(net_name);

  // make sure the net name is a valid file name
  std::replace(net_name_copy.begin(), net_name_copy.end(), '/', '_');
  std::replace(net_name_copy.begin(), net_name_copy.end(), '\\', '_');

  // take current local time
  time_t rawtime;
  std::time(&rawtime);
  struct tm timeinfo;
  localtime_r(&rawtime, &timeinfo);

  // and append it to the log file name, in a human-readable format
  std::string buf;
  buf.resize(30); // 15 should be enough, but apparently is too short.
  strftime(&buf[0], buf.size(), "%Y%m%d_%H%M%S", &timeinfo);
  auto datetime = buf.data();

  std::stringstream stream;
  stream << HTRACE_LOG_PATH_KEY << "=/tmp/htrace_" << net_name_copy << "_log_"
         << datetime << ";";
  stream << HTRACE_LOCAL_FILE_RCV_PATH_KEY << "=/tmp/htrace_" << net_name_copy
         << "_span_log_" << datetime << ";";
  stream << HTRACE_SPAN_RECEIVER_KEY << "=local.file;";
  stream << HTRACE_SAMPLER_KEY << "=always;";

  return stream.str();
}

class HTraceDAGNet : public DAGNetBase {
 public:
  HTraceDAGNet(const NetDef& net_def, Workspace* ws) : DAGNetBase(net_def, ws) {
    VLOG(1) << "Constructing HTrace DAG Net " << net_def.name();

    for (auto& worker : workers_) {
      std::thread::id worker_id = worker.get_id();
      std::stringstream stream;
      stream << "worker-scope-" << worker_id;
      htrace_worker_scope_map_[worker_id] = new htrace::Scope(
          htrace_tracer_, htrace_root_scope_.GetSpanId(), stream.str());
    }
  }

  bool Run() override {
    htrace::Scope run_scope(
        htrace_tracer_,
        htrace_root_scope_.GetSpanId(),
        "run-scope-" + caffe2::to_string(run_count_++));
    return DAGNetBase::Run();
  }

  ~HTraceDAGNet() {
    VLOG(1) << "Closing all htrace scopes for workers";
    for (const auto& kv : htrace_worker_scope_map_) {
      delete kv.second;
    }
  }

 protected:
  bool RunAt(const std::vector<int>& chain) override {
    std::thread::id thread_id = std::this_thread::get_id();
    htrace::Scope* worker_scope = htrace_worker_scope_map_[thread_id];

    bool success = true;
    for (const auto idx : chain) {
      auto def = operator_nodes_[idx].operator_->def();
      const string& print_name =
          (def.name().size() ? def.name() : (def.output_size() ? def.output(0)
                                                               : "NO_OUTPUT"));
      const string& op_type = def.type();

      htrace::Scope operator_scope(
          htrace_tracer_,
          worker_scope->GetSpanId(),
          "#" + caffe2::to_string(idx) + " (" + print_name + ", " + op_type +
              ")");
      success &= operator_nodes_[idx].operator_->Run();
    }
    return success;
  }

  htrace::Conf htrace_conf_{FLAGS_caffe2_htrace_conf.empty()
                                ? defaultHTraceConf(name_)
                                : FLAGS_caffe2_htrace_conf};
  htrace::Tracer htrace_tracer_{"htrace-tracer", htrace_conf_};
  htrace::Sampler htrace_sampler_{&htrace_tracer_, htrace_conf_};
  htrace::Scope htrace_root_scope_{htrace_tracer_,
                                   htrace_sampler_,
                                   "root-scope"};
  CaffeMap<std::thread::id, htrace::Scope*> htrace_worker_scope_map_;
  int run_count_ = 0;
};

REGISTER_NET(htrace_dag, HTraceDAGNet);

} // namespace
} // namespace caffe2
