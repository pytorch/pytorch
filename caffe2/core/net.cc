#include "caffe2/core/net.h"
#include "caffe2/core/net_simple.h"

#include <set>
#include <unordered_map>
#include <unordered_set>

#include "caffe2/core/operator.h"
#include "caffe2/core/timer.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/utils/proto_utils.h"
#include "caffe2/utils/string_utils.h"

CAFFE2_DEFINE_string(
    caffe2_override_executor,
    "",
    "Comma-separated list of executor overrides");

namespace caffe2 {

CAFFE_DEFINE_REGISTRY(
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

namespace {
const std::string kSimpleNet = "simple";

std::vector<NetObserverCreator>* GetNetObserverCreators() {
  static std::vector<NetObserverCreator> creators;
  return &creators;
}

void checkExecutorOverride(std::string& net_type) {
  auto executors = caffe2::split(',', FLAGS_caffe2_override_executor);
  CAFFE_ENFORCE(
      executors.size() % 2 == 0, "Invalid override executors flag value");
  std::unordered_map<std::string, std::string> overrides;
  for (auto idx = 0; idx < executors.size() - 1; idx += 2) {
    overrides[executors[idx]] = executors[idx + 1];
  }
  if (overrides.count(net_type)) {
    LOG(INFO) << "Overrode net type '" << net_type << "' with '"
              << overrides[net_type] << "'";
    net_type = overrides[net_type];
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
  if (net_def->has_type()) {
    net_type = net_def->type();
  } else {
    // By default, we will return a simple network that just runs all operators
    // sequentially.
    net_type = kSimpleNet;
  }
  if (!FLAGS_caffe2_override_executor.empty()) {
    checkExecutorOverride(net_type);
  }
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

std::shared_ptr<TaskThreadPool> ExecutorHelper::GetPool(
    const DeviceOption& /* unused */) const {
  CAFFE_THROW("Not implemented");
}

} // namespace caffe2
