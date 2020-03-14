#include "caffe2/caffe2/fb/predictor/Transforms.h"


namespace caffe2 {

namespace {
string
NextBlob(const Workspace& ws, const string& prefix, int max_tries = 1000000) {
  for (int i = 0; i < max_tries; ++i) {
    std::stringstream stream;
    stream << prefix;
    if (i) {
      stream << '_' << i;
    }
    if (!ws.HasBlob(stream.str())) {
      return stream.str();
    }
  }
  CAFFE_THROW("Failed to find a new blob name");
  return "";
}

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

/*
Tests if it is valid to simply rename "from" to "to" after start (at or after
start) ignoring in-place ops of operator types in ignoreTypes. There is a set of
ignoreTypes instead of a single ignoreType for uses cases where multiple
op-types will be in-placed; in such cases this function will give a more optimal
answer if it can consider all types at the same time.
*/
bool CanRenameForwards(
    const string& from,
    const string& to,
    const std::shared_ptr<NetDef>& net,
    const std::set<string>& netOutputs,
    const std::set<string>& ignoreTypes,
    int start) {
  bool redefined_to = false;
  for (int i = start; i < net->op_size(); i++) {
    auto op = net->op(i);
    bool uses_from = HasInput(from, op);

    // If there's a use of "from" after a redefine of "to" then we can't rename
    // this "from" to "to" b/c it will use the wrong "to" (this op's "to"
    // instead of the "to" that would be produced by renaming ops[start-1] to
    // output "to" to "from")
    if (redefined_to && uses_from) {
      VLOG(7) << "CanRenameForwards " << to << " is redefined before " << from
              << " is used. Cannot rename from " << from << " to " << to;
      return false;
    }
    // If "to" is redefined then we have to be careful of in-placing this op or
    // blocking future uses of "from"
    if (HasOutput(to, op)) {
      // If this op also uses "from", then renaming "from" to "to" will make
      // this op inplace
      if (uses_from) {
        // In-placing of this op is not allowed
        if (!ignoreTypes.count(op.type())) {
          VLOG(7) << "CanRenameForwards detected in-placing of op of type "
                  << op.type();
          return false;
        }
        VLOG(7) << "CanRenameForwards will make " << op.DebugString()
                << " in-place, but this is in the okay-types-to-inplace "
                << "whitelist";
      }
      // This op won't be inplaced (or it's okay if it is) but we still have to
      // watch out for future uses of "from"
      redefined_to = true;
      VLOG(7) << "CanRenameForwards " << to << " is redefined at " << i;
    }
    // If this op redefines "from" then renaming will stop with the inputs of
    // this op. Since we haven't found any problems with renaming, it's okay to
    // rename. Note that we don't need to check if "from" is a network output,
    // as this op will produce "from"
    if (HasOutput(from, op)) {
      VLOG(7) << "CanRenameForwards found another op making " << from
              << " so it's fine to rename " << from << " to " << to
              << " in earlier ops of the net.";
      return true;
    }
  }
  // We reached the end of the ops. There have been no redefinitions of "from",
  // so if "from" is needed in the network outputs then we can't rename it
  return !netOutputs.count(from);
}

bool CanRenameBackwards(
    const string& from,
    const string& to,
    const std::shared_ptr<NetDef>& net,
    const std::set<string>& netInputs,
    const std::set<string>& netOutputs,
    const std::set<string>& ignoreTypes,
    int end) {
  for (int i = end; i >= 0; i--) {
    auto op = net->op(i);
    // If this op defines "to", then all ops after this point will use this op's
    // "to" instead of the producer-of-"from"s, so the producer can't be renamed
    // to produce "to" instead of "from"
    // FUTURE_POSSIBILITY: We might be able to rename this op to not produce to
    if (HasOutput(to, op)) {
      VLOG(7) << "CanRenameBackwards " << to << " is defined after " << from
              << " is. Cannot rename.";
      return false;
    }
    // Because of the previous question, we know that no op in between this op
    // and end has "to" as an output, so it is impossible to make any of them
    // in-place by adding "to" as an input to any of them

    // If we find the producer of "from", then we will stop renaming backwards
    // (we won't rename this ops inputs)
    if (HasOutput(from, op)) {
      // If this op has "to" as an input, then renaming "from" to "to" will
      // in-place this op
      if (HasInput(to, op)) {
        if (!ignoreTypes.count(op.type())) {
          // In future, you could maybe check if "to" could be renamed to a
          // brand new unique blob name
          VLOG(7) << "CanRenameBackwards will in-place producer of " << from;
          return false;
        }
        VLOG(7) << "CanRenameBackwards will in-place the producer of " << from
                << " but this is okay because it's in our in-placeable "
                << "whitelist";
      }
      // This op won't be in-placed (or it will but it's still okay), but we
      // have to check the forwards logic too
      // TODO why? what's the specific case again?
      VLOG(7) << "CanRenameBackwards found the parent of " << from
              << ". Recursively testing if we can rename the parent.";
      return CanRenameForwards(from, to, net, netOutputs, ignoreTypes, i + 1);
    }
    // After this point, inputs of this op may be renamed

    // If this blob uses "to", then renaming the producer of "from" to produce
    // "to" will interfere with this op. Technically it will only interfere if
    // the producer of "from" would overwrite this op's "to", but if that wasn't
    // the case then there's some op that produces "to" after the producer of
    // "from", and this will be caught in the first if (HasOutput(to, op))
    if (HasInput(to, op)) {
      VLOG(7) << "CanRenameBackwards " << to << " is used after " << from
              << " is defined. Cannot rename.";
      return false;
    }
  }
  // Found no parent, so must be a network output. We cannot rename it
  CAFFE_ENFORCE(netInputs.count(from));
  VLOG(7) << "CanRenameBackwards " << from << " is a network input. Cannot "
          << "rename.";
  return false;
}

void RenameInputs(const string& from, const string& to, OperatorDef* def) {
  VLOG(6) << "RenameInputs(from=" << from << ", to=" << to << ", "
          << def->DebugString() << ")";
  for (int i = 0; i < def->input_size(); i++) {
    if (def->input(i) == from) {
      *def->mutable_input(i) = to;
    }
  }
}

void RenameOutputs(const string& from, const string& to, OperatorDef* def) {
  VLOG(6) << "RenameOutputs(from=" << from << ", to=" << to << ", "
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
  // This does NOT continue through in-place ops
  VLOG(4) << "RenameInputsInChildren(from=" << from << ", to=" << to;
  for (int j = pidx + 1; j < net->op_size(); j++) {
    if (HasInput(from, net->op(j))) {
      RenameInputs(from, to, net->mutable_op(j));
    }
    // If any child op redefines from, then future ops no longer use this op's
    // (at j) version of from
    if (HasOutput(from, net->op(j))) {
      return;
    }
  }
}

} // namespace

void InPlaceOps(const InferenceGraph& graph, const std::string& op_type) {
  int num_inplaced = 0;
  auto net = graph.predict_net_def;

  // Collect blob names that we can never rename
  std::set<string> netInputs(
      graph.parameter_names.begin(), graph.parameter_names.end());
  netInputs.insert(graph.input_names.begin(), graph.input_names.end());
  std::set<string> netOutputs(
      graph.output_names.begin(), graph.output_names.end());

  // In-place ops greedily in a forward manner
  for (int i = 0; i < net->op_size(); i++) {
    OperatorDef op = net->op(i);

    // Only inplace the requested
    if (op.type() != op_type) {
      VLOG(2) << "InPlaceObs: Type is " << op_type << ". Not in-placing";
      continue;
    }

    if (op.input_size() != 1 || op.output_size() != 1) {
      LOG(ERROR) << "InPlaceOps only supports ops with exactly 1 output "
                 << "and exactly 1 input. Skipping op " << op.DebugString();
      continue;
    }

    const string& in = op.input(0);
    const string& out = op.output(0);

    // If it's already in place then let's not do any more work
    if (in == out) {
      continue;
    }

    // Otherwise check if we can rename things
    bool can_rename_forwards =
        CanRenameForwards(out, in, net, netOutputs, {op_type}, i + 1);

    // If renaming is impossible (or complicated) then skip this op
    if (!can_rename_forwards &&
        !CanRenameBackwards(
            in, out, net, netInputs, netOutputs, {op_type}, i - 1)) {
      VLOG(2) << "InPlaceOps: Complicated or impossible remove for op: "
              << op.DebugString();
      continue;
    }
    VLOG(2) << "InPlaceOps will inplace " << op.DebugString();
    num_inplaced++;

    // Handle renaming
    if (can_rename_forwards) {
      // Rename out to in
      VLOG(3) << "InPlaceOps can rename in children from " << out << " to "
              << in;
      RenameInputsInChildren(out, in, net, i);

    } else {
      // Since out is an output of the network, we must rename in parents of op
      VLOG(3) << "InPlaceOps must find parent that produced " << in
              << " and rename in all of its children from " << in << " to "
              << out;
      for (int pidx = i - 1; pidx >= 0; pidx--) {
        if (HasOutput(in, net->op(pidx))) {
          VLOG(5) << "InPlaceOps found parent is "
                  << net->op(pidx).DebugString();
          RenameOutputs(in, out, net->mutable_op(pidx));
          RenameInputsInChildren(in, out, net, pidx);
          break;
        }
      }
    }
  } // For every op
  VLOG(1) << "InPlaceOps(" << op_type << ") renamed " << num_inplaced << " ops";
}

void RemoveOpsByType(const InferenceGraph& graph, const std::string& op_type) {
  int num_removed = 0;
  auto net = graph.predict_net_def;

  // Rename all the ops we want to delete
  InPlaceOps(graph, op_type);

  // Now the only ops we can delete are inplaced ones
  for (int i = 0; i < net->op_size(); i++) {
    OperatorDef op = net->op(i);

    // Only remove ops of the requested type
    if (op.type() != op_type) {
      VLOG(2) << "RemoveOpsByType: Type is " << op_type << ". Not removing";
      continue;
    }

    if (op.input_size() != 1 || op.output_size() != 1) {
      LOG(ERROR) << "RemoveOpsByType only supports ops with exactly 1 output "
                 << "and exactly 1 input. Skipping op " << op.DebugString();
      continue;
    }

    const string& in = op.input(0);
    const string& out = op.output(0);

    // If the op is in-place then we can always delete it
    if (in == out) {
      VLOG(1) << "RemoveOpsByType(" << op_type << ") deleting inplace op";
      net->mutable_op()->erase(net->mutable_op()->begin() + i);
      i--;
      num_removed++;
    } else {
      VLOG(2) << "RemoveOpsByType(" << op_type << ") can't delete.";
    }
  } // For every op
  VLOG(1) << "RemoveOpsByType(" << op_type << ") removed " << num_removed
          << " ops";
}

} // namespace caffe2
