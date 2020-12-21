#include "caffe2/core/net.h"
#include "caffe2/core/net_simple.h"

#include <set>
#include <unordered_map>
#include <unordered_set>

#include "caffe2/core/init.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/timer.h"
#include "caffe2/proto/caffe2_pb.h"
#include "caffe2/utils/proto_utils.h"
#include "caffe2/utils/string_utils.h"

C10_DEFINE_string(
    caffe2_override_executor,
    "",
    "Comma-separated list of executor overrides");

namespace caffe2 {

C10_DEFINE_REGISTRY(
    NetRegistry,
    NetBase,
    const std::shared_ptr<const NetDef>&,
    Workspace*);

NetBase::NetBase(
    const std::shared_ptr<const NetDef>& def,
    Workspace* /* unused */)
    : external_input_(
          def->external_input().begin(),
          def->external_input().end()),
      external_output_(
          def->external_output().begin(),
          def->external_output().end()),
      name_(def->name()),
      net_def_(def) {
  static GlobalInitIsCalledGuard guard;
  C10_LOG_API_USAGE_ONCE("caffe2.net.create");
  // Check that node_name is empty for all ops
  for (const OperatorDef& op : def->op()) {
    if (op.has_device_option()) {
      CAFFE_ENFORCE(
          !op.device_option().has_node_name(),
          "node_name must be empty for all operators at execution time.");
    }
  }

  // Go through the operators and make sure that blobs are correctly made.
  std::set<string> known_blobs(
      external_input_.begin(), external_input_.end());
  std::set<string> remaining_output(
      external_output_.begin(), external_output_.end());
  for (const auto& blob : known_blobs) {
    remaining_output.erase(blob);
  }
  for (const OperatorDef& op : def->op()) {
    for (const string& in : op.input()) {
      if (!known_blobs.count(in)) {
        if (external_input_.size()) {
          CAFFE_THROW(
              "op ",
              op.type(),
              ": Source for input ",
              in,
              " is unknown for net ",
              def->name(),
              ", operator ",
              ProtoDebugString(op));
        } else {
          // If we are not declaring input and output, we will simply VLOG it
          // for debugging purposes.
          VLOG(1) << "op " << op.type() << ": input " << in << " is unknown.";
        }
      }
    }
    for (const string& out : op.output()) {
      known_blobs.insert(out);
      remaining_output.erase(out);
    }
  }
  // Finally, check if all declared outputs are being created.
  CAFFE_ENFORCE(
      remaining_output.size() == 0,
      "Some of the blobs are declared as output but never produced by the "
      "net ",
      def->name(),
      ", the first one is ",
      *remaining_output.begin());
}

bool NetBase::RunAsync() {
  for (auto& op : GetOperators()) {
    op->ResetEvent();
  }
  return DoRunAsync();
}

void NetBase::Cancel() {
  for (auto& op : GetOperators()) {
    op->Cancel();
  }
}

namespace {
const std::string kSimpleNet = "simple";

std::vector<NetObserverCreator>* GetNetObserverCreators() {
  static std::vector<NetObserverCreator> creators;
  return &creators;
}

const std::unordered_map<std::string, std::string>& defaultOverrides() {
  // redirecting legacy net types to async_scheduling (except for 'simple');
  // async_scheduling checks net type for backward compatibility
  static const std::unordered_map<std::string, std::string> overrides = {
      {"dag", "async_scheduling"},
      {"prof_dag", "async_scheduling"},
      {"async_dag", "async_scheduling"},
      {"async_polling", "async_scheduling"},
      {"async_simple", "simple"}, // "async_simple" impl has been removed.
      {"rnn", "simple"}, // "rnn" impl has been removed.
  };
  return overrides;
}

void ApplyPotentialExecutorOverride(std::string* net_type) {
  auto executors = caffe2::split(',', FLAGS_caffe2_override_executor);
  CAFFE_ENFORCE(
      executors.size() % 2 == 0, "Invalid override executors flag value");
  std::unordered_map<std::string, std::string> overrides;
  for (const auto& kv : defaultOverrides()) {
    overrides[kv.first] = kv.second;
  }
  for (size_t idx = 0; idx < executors.size(); idx += 2) {
    overrides[executors[idx]] = executors[idx + 1];
  }
  if (overrides.count(*net_type)) {
    VLOG(1) << "Overrode net type '" << *net_type << "' with '"
            << overrides[*net_type] << "'";
    *net_type = overrides[*net_type];
  }
}

} // namespace

void AddGlobalNetObserverCreator(NetObserverCreator creator) {
  GetNetObserverCreators()->push_back(creator);
  VLOG(1) << "Have set a custom GlobalNetObserverCreator";
}

void ClearGlobalNetObservers() {
  GetNetObserverCreators()->clear();
  VLOG(1) << "All net observers cleared";
}

unique_ptr<NetBase> CreateNet(const NetDef& net_def, Workspace* ws) {
  std::shared_ptr<NetDef> tmp_net_def(new NetDef(net_def));
  return CreateNet(tmp_net_def, ws);
}

unique_ptr<NetBase> CreateNet(
    const std::shared_ptr<const NetDef>& net_def,
    Workspace* ws) {
  std::string net_type;
  if (net_def->has_type() && !net_def->type().empty()) {
    net_type = net_def->type();
  } else {
    // By default, we will return a simple network that just runs all operators
    // sequentially.
    net_type = kSimpleNet;
  }
  ApplyPotentialExecutorOverride(&net_type);
  unique_ptr<NetBase> net = NetRegistry()->Create(net_type, net_def, ws);

  VLOG(1) << "Adding a global observer to a net";
  if (net) {
    auto* observer_creators = GetNetObserverCreators();
    for (auto& creator : *observer_creators) {
      net->AttachObserver(creator(net.get()));
    }
  }
  return net;
}

TaskThreadPoolBase* ExecutorHelper::GetPool(
    const DeviceOption& /* unused */) const {
  CAFFE_THROW("Not implemented");
}

std::vector<OperatorBase*> ExecutorHelper::GetOperators() const {
  CAFFE_THROW("Not implemented");
}

int ExecutorHelper::GetNumWorkers() const {
  CAFFE_THROW("Not implemented");
}

// benchmark an individual run so that we can FeedBlobs with new inputs
// no warmup
// return time taken in microseconds
float NetBase::TEST_Benchmark_One_Run() {
  Timer timer;
  CAFFE_ENFORCE(Run(), "Run has failed.");
  return timer.MicroSeconds();
}

std::vector<float> NetBase::TEST_Benchmark(
    const int warmup_runs,
    const int main_runs,
    const bool run_individual) {
  LOG(INFO) << "Starting benchmark, running warmup runs";
  CAFFE_ENFORCE(
      warmup_runs >= 0,
      "Number of warm up runs should be non negative, provided ",
      warmup_runs);
  for (int run_idx = 0; run_idx < warmup_runs; ++run_idx) {
    CAFFE_ENFORCE(Run(), "Warmup run ", run_idx, " has failed");
  }

  LOG(INFO) << "Running main runs";
  CAFFE_ENFORCE(
      main_runs >= 0,
      "Number of main runs should be non negative, provided ",
      main_runs);

  Timer timer;
  for (int run_idx = 0; run_idx < main_runs; ++run_idx) {
    CAFFE_ENFORCE(Run(), "Main run ", run_idx, " has failed");
  }
  auto millis = timer.MilliSeconds();
  LOG(INFO) << "Main runs finished. Milliseconds per iter: "
            << millis / main_runs
            << ". Iters per second: " << 1000.0 * main_runs / millis;

  if (run_individual) {
    LOG(INFO) << "Net does not support per-op benchmark; "
                 "to run it, switch to a simple net type";
  }
  return std::vector<float>{millis / main_runs};
}

} // namespace caffe2
