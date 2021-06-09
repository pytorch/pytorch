#include "caffe2/operators/create_scope_op.h"

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
C10_DEFINE_bool(
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(CreateScope, CreateScopeOp<CPUContext>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
SHOULD_NOT_DO_GRADIENT(CreateScope);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(CreateScope).NumInputs(0).NumOutputs(1).SetDoc(R"DOC(
'CreateScope' operator initializes and outputs empty scope that is used
by Do operator to store local blobs
    )DOC");

template <>
bool HasScopeOp<CPUContext>::RunOnDevice() {
  const auto& ws_stack = OperatorBase::Input<detail::WorkspaceStack>(0);

  auto* output = Output(0, {1}, at::dtype<bool>());
  bool* output_value = output->template mutable_data<bool>();
  *output_value = !ws_stack.empty();
  return true;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(HasScope, HasScopeOp<CPUContext>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
SHOULD_NOT_DO_GRADIENT(HasScope);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(HasScope).NumInputs(1).NumOutputs(1).SetDoc(R"DOC(
Checks whether scope blob has any saved scopes left
    )DOC");

} // namespace caffe2
