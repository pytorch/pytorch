#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/OpsAlreadyMovedToC10.h>
#include <torch/csrc/jit/runtime/operator.h>

namespace torch {
namespace jit {
namespace {

class RegistrationListener final : public c10::OpRegistrationListener {
public:
  void onOperatorRegistered(const c10::OperatorHandle& op) override {
    if (at::is_aten_op_and_unboxing_is_not_handled_by_c10_yet(op.schema().operator_name())) {
      // register_aten_ops.cpp registers the jit unboxing wrapper for this op, no need to do anything here.
    } else {
      torch::jit::registerOperator(Operator(op));
    }
  }

  void onOperatorDeregistered(const c10::OperatorHandle& op) override {
    // TODO Do something like torch::jit::deregisterOperator(op.schema());
  }
};

struct Registerer final {
  Registerer() {
    // this immediately calls the listener on all existing ops,
    // and calls it in future whenever a new op is registered
    c10::Dispatcher::singleton().addRegistrationListener(
      std::make_unique<RegistrationListener>()
    );
  }
};

Registerer& registerer() {
  static Registerer registerer;
  return registerer;
}

// global instance to run its constructor on startup
Registerer& dummy = registerer();

} // namespace

void ensure_c10_registerer_defined() {
  registerer();
}

} // namespace jit
}
