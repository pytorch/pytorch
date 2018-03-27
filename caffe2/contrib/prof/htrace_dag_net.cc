#include <htrace.hpp>

#include "caffe2/contrib/prof/htrace_conf.h"
#include "caffe2/core/net_dag.h"
#include "caffe2/core/operator.h"

namespace caffe2 {
namespace {

class HTraceDAGNet : public DAGNetBase {
 public:
  HTraceDAGNet(const std::shared_ptr<const NetDef>& net_def, Workspace* ws)
      : DAGNetBase(net_def, ws) {
    VLOG(1) << "Constructing HTrace DAG Net " << net_def->name();

    for (auto& worker : workers_) {
      std::thread::id worker_id = worker.get_id();
      std::stringstream stream;
      stream << "worker-scope-" << worker_id;
      htrace_worker_scope_map_[worker_id] = std::make_shared<htrace::Scope>(
          htrace_tracer_, htrace_root_scope_.GetSpanId(), stream.str());
    }
  }

  bool SupportsAsync() override {
    return false;
  }

  ~HTraceDAGNet() {
    VLOG(1) << "Closing all htrace scopes for workers";

    // Due to the implementation of htrace,
    // we need to make sure we delete the scopes in order.
    // Simply calling map.clear() may not preserve the order.
    auto iter = htrace_worker_scope_map_.begin();
    while (iter != htrace_worker_scope_map_.end()) {
      iter = htrace_worker_scope_map_.erase(iter);
    }
  }

 protected:
  bool DoRunAsync() override {
    htrace::Scope run_scope(
        htrace_tracer_,
        htrace_root_scope_.GetSpanId(),
        "run-scope-" + caffe2::to_string(run_count_++));
    return DAGNetBase::DoRunAsync();
  }

  bool RunAt(int /* unused */, const std::vector<int>& chain) override {
    std::thread::id thread_id = std::this_thread::get_id();
    auto worker_scope = htrace_worker_scope_map_[thread_id];

    bool success = true;
    for (const auto idx : chain) {
      const auto& op = operator_nodes_[idx].operator_;
      const auto& def = op->debug_def();
      const string& print_name =
          (def.name().size()
               ? def.name()
               : (op->OutputSize() ? def.output(0) : "NO_OUTPUT"));
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

  htrace::Conf htrace_conf_{defaultHTraceConf(name_)};
  htrace::Tracer htrace_tracer_{"htrace-tracer", htrace_conf_};
  htrace::Sampler htrace_sampler_{&htrace_tracer_, htrace_conf_};
  htrace::Scope htrace_root_scope_{htrace_tracer_,
                                   htrace_sampler_,
                                   "root-scope"};
  std::map<std::thread::id, std::shared_ptr<htrace::Scope>>
      htrace_worker_scope_map_;
  int run_count_ = 0;
};

REGISTER_NET(htrace_dag, HTraceDAGNet);

} // namespace
} // namespace caffe2
