#include "caffe2/caffe2/fb/predictor/Transforms.h"
#include "caffe2/onnx/onnx_exporter.h"

#include <unordered_set>

namespace caffe2 {

namespace {
bool HasInput(const string& blob, const OperatorDef& op) {
  for (auto in : op.input()) {
    if (blob == in) {
      return true;
    }
  }
  return false;
}

bool HasOutput(const string& blob, const OperatorDef& op) {
  for (auto out : op.output()) {
    if (blob == out) {
      return true;
    }
  }
  return false;
}

void RenameInputs(const string& from, const string& to, OperatorDef* def) {
  VLOG(2) << "RenameInputs (from=" << from << ", to=" << to << ", "
          << def->DebugString() << ")";
  for (int i = 0; i < def->input_size(); i++) {
    if (def->input(i) == from) {
      *def->mutable_input(i) = to;
    }
  }
}

void RenameOutputs(const string& from, const string& to, OperatorDef* def) {
  VLOG(2) << "RenameOutputs (from=" << from << ", to=" << to << ", "
          << def->DebugString() << ")";
  for (string& output : *def->mutable_output()) {
    if (output == from) {
      output = to;
    }
  }
}

void RenameInputsInChildren(
    const string& from,
    const string& to,
    std::shared_ptr<caffe2::NetDef> net,
    int pidx) {
  VLOG(2) << "RenameInputsInChildren (from=" << from << ", to=" << to << ")";
  for (int j = pidx + 1; j < net->op_size(); j++) {
    if (HasInput(from, net->op(j))) {
      RenameInputs(from, to, net->mutable_op(j));
    }
  }
}

int RenameOutputInParents(
    const std::string& from,
    const std::string& to,
    std::shared_ptr<caffe2::NetDef> net,
    int idx) {
  VLOG(2) << "RenameOutputInParents (from=" << from << ", to=" << to << ")";
  for (int i = 0; i < idx; i++) {
    if (HasOutput(from, net->op(i))) {
      // There can only be 1 producer of a particular output in SSA form
      RenameOutputs(from, to, net->mutable_op(i));
      // return the index of the op that produces from
      return i;
    }
  }
  return -1;
}

} // namespace

// Inputs to the model are strictly read-only, and can't be overwritten in the
// net. This is enforced in BlackBoxPredictor. Otherwise, the algorithm would
// break.
void RemoveOpsByType(InferenceGraph& graph, const std::string& op_type) {
  int num_removed = 0;
  std::shared_ptr<NetDef> net = graph.predict_net_def;

  std::unordered_set<std::string> inputs(
      graph.input_names.begin(), graph.input_names.end());
  std::unordered_set<std::string> outputs(
      graph.output_names.begin(), graph.output_names.end());

  // add external_outputs to net as they're necessary to correctly do ssa
  // rewriting
  if (!graph.predictor_net_ssa_rewritten) {
    net->mutable_external_output()->Clear();
    for (const auto& o : graph.output_names) {
      net->add_external_output(o);
    }
    onnx::SsaRewrite(nullptr, net.get());
    net->mutable_external_output()->Clear();
    graph.predictor_net_ssa_rewritten = true;
  }

  // Inplace ops
  for (int i = 0; i < net->op_size(); i++) {
    OperatorDef* op = net->mutable_op(i);
    if (op->type() != op_type) {
      VLOG(2) << "InplaceOps(" << op_type << ") skipping op: \n"
              << op->DebugString();
      continue;
    }
    if (op->input_size() != 1 || op->output_size() != 1) {
      VLOG(2) << "InplaceOps(" << op_type
              << ") only supports ops with exactly 1 output "
              << "and exactly 1 input. Skipping op: \n"
              << op->DebugString();
      continue;
    }

    // use actual copy because op->input/output may change
    const std::string in = op->input(0);
    const std::string out = op->output(0);

    if (in == out) {
      // this case can still exist when in/out is in the predict_net's outputs
      continue;
    }

    if (outputs.count(out) != 0) {
      if (inputs.count(in) == 0 && outputs.count(in) == 0) {
        // Rename all apperances (regardless of inputs/outputs) of in (if not
        // in inputs) to out, when out is guaranteed to be produced a parent
        // op
        int idx = RenameOutputInParents(in, out, net, i);
        if (idx >= 0) {
          VLOG(2) << "InplaceOps(" << op_type << ") inplacing op:\n"
                  << op->DebugString();
          RenameInputsInChildren(in, out, net, idx);
          op->set_input(0, out);
          // Redo inplacing ops starting from idx
          i = idx;
        }
      } else {
        VLOG(2) << "InplaceOps(" << op_type << ") skipping op: \n"
                << op->DebugString();
      }
    } else {
      // Rename all apperances of out to in
      VLOG(2) << "InplaceOps(" << op_type << ") inplacing op:\n"
              << op->DebugString();
      RenameInputsInChildren(out, in, net, i);
      op->set_output(0, in);
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
  LOG(INFO) << "RemoveOpsByType(" << op_type << ") removed " << num_removed
            << " ops";
}

} // namespace caffe2
