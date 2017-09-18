#include "caffe2/operators/create_scope_op.h"

CAFFE2_DEFINE_bool(
    caffe2_workspace_stack_debug,
    false,
    "Enable debug checks for CreateScope's workspace stack");

namespace caffe2 {
CAFFE_KNOWN_TYPE(detail::WorkspaceStack);

template <>
bool CreateScopeOp<CPUContext>::RunOnDevice() {
  auto* ws_stack = OperatorBase::Output<detail::WorkspaceStack>(0);
  ws_stack->clear();
  return true;
}

REGISTER_CPU_OPERATOR(CreateScope, CreateScopeOp<CPUContext>);

SHOULD_NOT_DO_GRADIENT(CreateScope);

OPERATOR_SCHEMA(CreateScope).NumInputs(0).NumOutputs(1).SetDoc(R"DOC(
'CreateScope' operator initializes and outputs empty scope that is used
by Do operator to store local blobs
    )DOC");

} // namespace caffe2
