#include "caffe2/operators/do_op.h"

#include "caffe2/operators/create_scope_op.h"

namespace caffe2 {

template <>
bool DoOp<CPUContext>::RunOnDevice() {
  auto* ws_stack =
      OperatorBase::Output<detail::WorkspaceStack>(OutputSize() - 1);
  std::shared_ptr<Workspace> net_workspace;
  if (is_gradient_op_) {
    net_workspace = ws_stack->popGradientWorkspace(parent_ws_, blob_bindings_);
  } else {
    net_workspace = ws_stack->pushForwardWorkspace(parent_ws_, blob_bindings_);
  }
  CAFFE_ENFORCE(net_workspace, "Failed to initialize Do op workspace");

  // TODO(iliacher): figure how to reuse existing net with a new workspace
  auto* net = net_workspace->GetNet(net_def_.name());
  if (!net) {
    net = net_workspace->CreateNet(net_def_, true);
  }
  CAFFE_ENFORCE(net, "Failed to initialize subnet");
  return net->Run();
}

REGISTER_CPU_OPERATOR(Do, DoOp<CPUContext>);

OPERATOR_SCHEMA(Do)
    .NumInputs(1, INT_MAX)
    .NumOutputs(1, INT_MAX)
    .SetDoc(R"DOC(
'Do' control operator, creates a new workspace and executes a subnet in it.
Last blob in the output list holds pointer to the op workspace. In case of
gradient Do, last blob in the input list should should the pointer to the
forward Do's workspace. Arguments 'inner_blobs' and 'outer_blobs_idx'
provide a mapping between selected inner blob names and corresponding outer blobs
indices.
    )DOC")
    .Arg("net", "Subnet with blob bindings")
    .Arg(
        "inner_blobs",
        "List of inner net blob names to bind to outer workspace")
    .Arg(
        "outer_blobs_idx",
        "Indices of corresponding outer workspace blobs, "
        "in order: operator inputs, operator outputs (skipping workspace blobs)")
    .Arg(
        "saved_fwd_blobs",
        "List of blobs from the forward Do operator workspace needed "
        "in backward pass, used in gradient Do operator")
    .AllowInplace([](int in, int out) -> bool { return true; });

} // namespace caffe2
