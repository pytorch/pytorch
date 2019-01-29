#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/csrc/jit/operator.h>

namespace torch {
namespace jit {
namespace {

// TODO This currently only handles tensors with requires_grad==False correctly.
//      It should also handle autograd.
Operator createOperatorFromC10(const c10::OperatorHandle& op) {
  return Operator(op.schema(), [op](Stack& stack) {
      const auto input_size = op.schema().arguments().size();
      const auto output_size = op.schema().returns().size();

      // unwrap tensor inputs from variable
      for (auto iter = stack.end() - input_size; iter != stack.end(); ++iter) {
        // TODO Remove the .defined() check once we don't have undefined tensors on the stack anymore (@wanchaol is working on this)
        if (iter->isTensor() && iter->toTensor().defined()) {
          at::Tensor tensor = std::move(*iter).toTensor();
          if (tensor.requires_grad()) {
            throw std::runtime_error("Autograd not yet supported for c10 ops.");
          }
          *iter = torch::autograd::Variable(std::move(tensor)).data();
        }
      }

      c10::Dispatcher::singleton().lookup(op, &stack).call(&stack);

      // wrap tensor outputs as variable
      for (auto iter = stack.end() - output_size; iter != stack.end(); ++iter) {
        if (iter->isTensor()) {
          *iter = torch::autograd::make_variable(std::move(*iter).toTensor());
        }
      }

      return 0;
  });
}

class RegistrationListener final : public c10::OpRegistrationListener {
public:
  void onOperatorRegistered(const c10::OperatorHandle& op) override {
    torch::jit::registerOperator(createOperatorFromC10(op));
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
      c10::guts::make_unique<RegistrationListener>()
    );
  }
};

// global instance to run its constructor on startup
Registerer registerer;

}
}
}
