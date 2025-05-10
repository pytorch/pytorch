#include <ATen/core/ATenOpList.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/record_function.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/operator.h>

namespace torch::jit {

namespace {

Operator createOperatorFromC10(const c10::OperatorHandle& op) {
  return Operator(op, [op](Stack& stack) { op.callBoxed(stack); });
}

class RegistrationListener final : public c10::OpRegistrationListener {
 public:
  void onOperatorRegistered(const c10::OperatorHandle& op) override {
    if (op.schema().name() == "aten::backward") {
      // aten::backward has a manual wrapper in register_prim_ops_fulljit.cpp.
      // We should not additionally export the c10 aten::backward op from
      // native_functions.yaml to JIT. This special handling is needed because
      // aten::backward requires AliasAnalysisKind::CONSERVATIVE but all ops
      // from native_functions.yaml get AliasAnalysisKind::FROM_SCHEMA.
      // TODO Find a better way to handle this.
      return;
    }
    torch::jit::registerOperator(createOperatorFromC10(op));
  }

  void onOperatorDeregistered(const c10::OperatorHandle& op) override {
    if (op.schema().name() == "aten::backward") {
      // see comment in onOperatorRegistered for why aten::backward is excluded
      return;
    }
    torch::jit::deregisterOperator(op.schema());
  }
};

struct Registerer final {
  // this immediately calls the listener on all existing ops,
  // and calls it in future whenever a new op is registered
  Registerer()
      : listenerRAII(c10::Dispatcher::singleton().addRegistrationListener(
            std::make_unique<RegistrationListener>())) {}
  c10::RegistrationHandleRAII listenerRAII;
};

Registerer& registerer() {
  static Registerer registerer;
  return registerer;
}

// global instance to run its constructor on startup
[[maybe_unused]] Registerer& dummy = registerer();

} // namespace

void ensure_c10_registerer_defined() {
  registerer();
}

} // namespace torch::jit
