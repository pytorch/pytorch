#include "caffe2/core/net_simple_refcount.h"
#include "caffe2/core/net.h"

#include <iostream>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include "caffe2/core/operator.h"
#include "caffe2/core/timer.h"
#include "caffe2/proto/caffe2_pb.h"
#include "caffe2/utils/proto_utils.h"

namespace caffe2 {

SimpleRefCountNet::SimpleRefCountNet(
    const std::shared_ptr<const NetDef>& net_def,
    Workspace* ws)
    : SimpleNet(net_def, ws) {
  VLOG(1) << "Constructing SimpleRefCountNet " << net_def->name();
  // Construct the "to delete" list.
  delete_list_.resize(net_def->op_size());

  std::map<string, int> last_consumed_at;
  std::set<string> created_by_me;
  // For each operator
  for (int idx = 0; idx < net_def->op_size(); ++idx) {
    const auto& op_def = net_def->op(idx);
    for (const string& in_name : op_def.input()) {
      last_consumed_at[in_name] = idx;
    }
    for (const string& out_name : op_def.output()) {
      created_by_me.insert(out_name);
    }
  }
  // We do not delete any operator that is not produced by the net, and
  // any operator that is marked as external_output. Any blob that is not
  // consumed won't be in the last_consumed_at map, so we don't need to
  // do anything special.
  for (auto& kv : last_consumed_at) {
    if (!created_by_me.count(kv.first)) {
      kv.second = -1;
    }
  }
  for (const string& name : net_def->external_output()) {
    last_consumed_at[name] = -1;
  }
  // Set up the delete list.
  for (auto& kv : last_consumed_at) {
    if (kv.second > 0) {
      delete_list_[kv.second].push_back(ws->GetBlob(kv.first));
      VLOG(1) << "NetSimpleRefCountNet: will delete " << kv.first
              << " at operator #" << kv.second;
    }
  }
}

bool SimpleRefCountNet::Run() {
  StartAllObservers();
  VLOG(1) << "Running net " << name_;
  for (auto op_id = 0U; op_id < operators_.size(); ++op_id) {
    auto& op = operators_[op_id];
    VLOG(1) << "Running operator " << op->debug_def().name() << "("
            << op->debug_def().type() << ").";
    bool res = op->Run();
    if (!res) {
      LOG(ERROR) << "Operator failed: " << ProtoDebugString(op->debug_def());
      return false;
    }
    for (Blob* blob : delete_list_[op_id]) {
      blob->Reset();
    }
  }
  StopAllObservers();
  return true;
}

REGISTER_NET(simple_refcount, SimpleRefCountNet);

} // namespace caffe2
