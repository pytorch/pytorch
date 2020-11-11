#include "caffe2/caffe2/fb/predictor/Transforms.h"
#include "caffe2/onnx/onnx_exporter.h"
#include "caffe2/utils/proto_utils.h"

#include <unordered_set>

namespace caffe2 {

namespace {
bool HasInput(const string& blob, const OperatorDef& op) {
  for (const auto& in : op.input()) {
    if (blob == in) {
      return true;
    }
  }
  return false;
}

bool HasOutput(const string& blob, const OperatorDef& op) {
  for (const auto& out : op.output()) {
    if (blob == out) {
      return true;
    }
  }
  return false;
}

void RewriteSubnetsForIfOp(
    const string& from,
    const string& to,
    OperatorDef* op) {
  ArgumentHelper helper(*op);
  Argument *then_arg = nullptr, *else_arg = nullptr;

  std::map<std::string, std::string> oldname_to_newname;
  oldname_to_newname[from] = to;

  if (helper.HasSingleArgumentOfType<NetDef>("then_net")) {
    then_arg = GetMutableArgument("then_net", false, op);
    onnx::rewriteSubnet(then_arg, oldname_to_newname);
  }
  if (helper.HasSingleArgumentOfType<NetDef>("else_net")) {
    else_arg = GetMutableArgument("else_net", false, op);
    onnx::rewriteSubnet(else_arg, oldname_to_newname);
  }
}

void RenameInputs(
    const string& from,
    const string& to,
    OperatorDef* def,
    int op_idx,
    std::unordered_map<std::string, std::unordered_set<int>>& children) {
  VLOG(2) << "RenameInputs (from=" << from << ", to=" << to << ", "
          << def->DebugString() << ")";
  for (int i = 0; i < def->input_size(); i++) {
    if (def->input(i) == from) {
      *def->mutable_input(i) = to;
      children[from].erase(op_idx);
      children[to].insert(op_idx);
    }
  }
  // Rename inputs in the subnets of If/AsyncIf op
  if (def->type() == "If" || def->type() == "AsyncIf") {
    RewriteSubnetsForIfOp(from, to, def);
  }
}

void RenameOutputs(
    const string& from,
    const string& to,
    OperatorDef* def,
    int op_idx,
    std::unordered_map<std::string, std::unordered_set<int>>& parents) {
  VLOG(2) << "RenameOutputs (from=" << from << ", to=" << to << ", "
          << def->DebugString() << ")";
  for (string& output : *def->mutable_output()) {
    if (output == from) {
      output = to;
      parents[from].erase(op_idx);
      parents[to].insert(op_idx);
    }
  }
  // Rename outputs in the subnets of If/AsyncIf op
  if (def->type() == "If" || def->type() == "AsyncIf") {
    RewriteSubnetsForIfOp(from, to, def);
  }
}

void RenameInputsInChildren(
    const string& from,
    const string& to,
    caffe2::NetDef* net,
    std::unordered_map<std::string, std::unordered_set<int>>& children) {
  VLOG(2) << "RenameInputsInChildren (from=" << from << ", to=" << to << ")";
  if (children.count(from) == 0) {
    return;
  }

  // make an temporary copy here because we're going to modify children
  for (int child : std::unordered_set<int>(children[from])) {
    RenameInputs(from, to, net->mutable_op(child), child, children);
  }
}

void RenameOutputInParents(
    const std::string& from,
    const std::string& to,
    caffe2::NetDef* net,
    std::unordered_map<std::string, std::unordered_set<int>>& parents) {
  VLOG(2) << "RenameOutputInParents (from=" << from << ", to=" << to << ")";
  if (parents.count(from) == 0) {
    return;
  }

  // make an temporary copy here because we're going to modify parents
  for (int parent : std::unordered_set<int>(parents[from])) {
    RenameOutputs(from, to, net->mutable_op(parent), parent, parents);
  }
}

bool FoundOpCandidate(
    const OperatorDef* op,
    int op_idx,
    const std::string& op_type,
    const std::unordered_set<std::string>& inputs,
    const std::unordered_set<std::string>& outputs,
    const std::unordered_map<std::string, std::unordered_set<int>>& parents,
    const std::unordered_map<std::string, std::unordered_set<int>>& children) {
  if (op->type() != op_type) {
    VLOG(2) << "InplaceOps(" << op_type << ") skipping op: \n"
            << op->DebugString();
    return false;
  }
  if (op->input_size() != 1 || op->output_size() != 1) {
    VLOG(2) << "InplaceOps(" << op_type
            << ") only supports ops with exactly 1 output "
            << "and exactly 1 input. Skipping op: \n"
            << op->DebugString();
    return false;
  }

  // use actual copy because op->input/output may change
  const std::string in = op->input(0);
  const std::string out = op->output(0);

  if (in == out) {
    // This case can still exist when in/out is in the predict_net's outputs.
    // The op is an inplace op already.
    return false;
  }

  // The following is to handle the special cases of inputs being overwritten
  // by ops in the net and then appear in outputs of the net
  if (outputs.count(out) == 0) {
    // Propagate input downwards
    // Make sure that after input is propagated down, it doesn't have parents
    // that comes after i but before the new child
    int earliest_child = INT_MAX;
    const auto& iter = children.find(out);
    if (iter != children.end()) {
      for (int child : iter->second) {
        earliest_child = std::min(earliest_child, child);
      }
    }
    if (earliest_child == INT_MAX) {
      return true;
    }
    const auto& iter2 = parents.find(in);
    if (iter2 != parents.end()) {
      for (int parent : iter2->second) {
        if (parent > op_idx && parent < earliest_child) {
          VLOG(2) << "InplaceOps(" << op_type << ") skipping op: \n"
                  << op->DebugString();
          return false;
        }
      }
    }
  } else {
    // Propagate output upwards
    if (inputs.count(in) != 0 || outputs.count(in) != 0) {
      // This is the case when the op is absolutely needed. It exists to serve
      // one and only one purpose, to copy from in to out where in is one of
      // the net's inputs or outputs and out is one of the net's outputs.
      VLOG(2) << "InplaceOps(" << op_type << ") skipping op: \n"
              << op->DebugString();
      return false;
    }
    // find latest parent of in
    int latest_parent = -1;
    const auto& iter = parents.find(in);
    if (iter != parents.end()) {
      for (int parent : iter->second) {
        latest_parent = std::max(latest_parent, parent);
      }
    }
    if (latest_parent == -1) {
      return false;
    }
    // Make sure that after output is propagated, it doesn't have children that
    // comes after its new parent, but before its previous parent
    const auto& iter2 = children.find(out);
    if (iter2 != children.end()) {
      for (int child : iter2->second) {
        if (child < op_idx && child > latest_parent) {
          VLOG(2) << "InplaceOps(" << op_type << ") skipping op: \n"
                  << op->DebugString();
          return false;
        }
      }
    }
  }

  return true;
}

} // namespace

// Conceptually it's a pretty easy process and consists of 3 steps:
// 1) SSA rewrite; 2) propagate inputs forwards; 3) propagate outputs
// backwards and then forwards again. However, because of model outputs
// which can't be overwritten during the SSA process, and the fact that
// inputs could be overwritten by ops and also appear in outputs, it adds
// a lot of extra complexity to handle these special cases. A lot of this
// extra complexity is handled in FoundOpCandidate.
void RemoveOpsByType(InferenceGraph& graph, const std::string& op_type) {
  int num_removed = 0;
  NetDef* net = graph.predict_net_def.get();
  for (auto& op : net->op()) {
    if (op.type() == "RecurrentNetwork") {
      LOG(INFO) << "RemoveOpsByType does not support RecurrentNetwork yet";
      return;
    }
  }

  std::unordered_set<std::string> inputs(
      graph.input_names.begin(), graph.input_names.end());
  std::unordered_set<std::string> outputs(
      graph.output_names.begin(), graph.output_names.end());

  if (!graph.predictor_net_ssa_rewritten) {
    net->mutable_external_output()->Clear();
    // add external_outputs to net as they're necessary to correctly do ssa
    // rewriting
    for (const auto& o : graph.output_names) {
      net->add_external_output(o);
    }
    onnx::SsaRewrite(nullptr, net);
    // clear external_outputs
    net->mutable_external_output()->Clear();
    graph.predictor_net_ssa_rewritten = true;
  }

  // construct parents/children graphs to facilitate graph traversal
  std::unordered_map<std::string, std::unordered_set<int>> parents, children;
  for (int i = 0; i < net->op_size(); i++) {
    OperatorDef* op = net->mutable_op(i);
    for (auto& in : op->input()) {
      children[in].insert(i);
    }
    for (auto& output : op->output()) {
      parents[output].insert(i);
    }
  }

  // Inplace ops. Step 1: propagate inputs downward
  for (int i = 0; i < net->op_size(); i++) {
    OperatorDef* op = net->mutable_op(i);
    if (!FoundOpCandidate(op, i, op_type, inputs, outputs, parents, children)) {
      continue;
    }
    const std::string in = op->input(0);
    const std::string out = op->output(0);
    if (outputs.count(out) == 0) {
      // Rename all apperances of out to in
      VLOG(2) << "InplaceOps(" << op_type << ") inplacing op:\n"
              << op->DebugString();
      RenameInputsInChildren(out, in, net, children);
      RenameOutputs(out, in, op, i, parents);
    }
  }

  // Step 2: propagate outputs upward
  for (int i = 0; i < net->op_size(); i++) {
    OperatorDef* op = net->mutable_op(i);
    if (!FoundOpCandidate(op, i, op_type, inputs, outputs, parents, children)) {
      continue;
    }
    const std::string in = op->input(0);
    const std::string out = op->output(0);
    if (outputs.count(out) != 0) {
      if (inputs.count(in) == 0 && outputs.count(in) == 0) {
        // Rename all apperances (regardless of inputs/outputs) of in (if not
        // in inputs) to out, when out is guaranteed to be produced a parent
        // op. With the parents/children graph which remembers all apprerances
        // of nodes (not just immediate parent/children), we don't need to
        // propagate the outputs back down again because those cases are already
        // handled by RenameOutputInParents and RenameInputsInChildren
        if (parents.count(in) > 0 && !parents[in].empty()) {
          RenameOutputInParents(in, out, net, parents);
          VLOG(2) << "InplaceOps(" << op_type << ") inplacing op:\n"
                  << op->DebugString();
          RenameInputsInChildren(in, out, net, children);
          RenameInputs(in, out, op, i, children);
        }
      }
    }
  }

  // Remove inplace ops
  int i = 0;
  while (i < net->op_size()) {
    OperatorDef op = net->op(i);
    if (op.type() == op_type && op.input_size() == 1 && op.output_size() == 1 &&
        op.input(0) == op.output(0)) {
      net->mutable_op()->erase(net->mutable_op()->begin() + i);
      num_removed++;
      VLOG(2) << "RemoveOpsByType(" << op_type << ") deleting inplace op: \n"
              << op.DebugString();
    } else {
      i++;
      VLOG(2) << "RemoveOpsByType(" << op_type << ") skipping op: \n"
              << op.DebugString();
    }
  }
  VLOG(2) << "RemoveOpsByType(" << op_type << ") removed " << num_removed
          << " ops";
}
} // namespace caffe2
