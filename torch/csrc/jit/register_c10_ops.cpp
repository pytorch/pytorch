#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/csrc/jit/operator.h>
#include <torch/csrc/jit/tracer.h>

namespace torch {
namespace jit {
namespace {

at::Tensor unwrap(at::Tensor&& tensor) {
  if (tensor.requires_grad()) {
    throw std::runtime_error("Autograd not yet supported for c10 ops.");
  }
  return torch::autograd::Variable(std::move(tensor)).data();
}

// TODO This currently only handles tensors with requires_grad==False correctly.
//      It should also handle autograd.
Operator createOperatorFromC10(const c10::OperatorHandle& op) {
  return Operator(op.schema(), [op](Stack& stack) {
      const auto input_size = op.schema().arguments().size();
      const auto output_size = op.schema().returns().size();

      Node* node = nullptr;

      // unwrap tensor inputs from variable
      for (auto iter = stack.end() - input_size; iter != stack.end(); ++iter) {
        // TODO Remove the .defined() check once we don't have undefined tensors on the stack anymore (@wanchaol is working on this)
        if (iter->isTensor() && iter->toTensor().defined()) {
          *iter = unwrap(std::move(*iter).toTensor());
        } else if (iter->isTensorList()) {
          for (auto& item : iter->toTensorList()->elements()) {
            item = unwrap(std::move(item));
          }
        }
      }

      if (jit::tracer::isTracing()) {
        auto symbol = Symbol::fromQualString(op.schema().name());
        const auto& graph = tracer::getTracingState()->graph;
        node = graph->create(symbol, 0);
        const auto& args = op.schema().arguments();
        int i = 0;
        for (auto iter = stack.end() - input_size; iter != stack.end();
             ++iter, ++i) {
          // TODO we need to refactor graph APIs (e.g., addInputs)
          // appropriately; after that, we can get rid of the giant if-else
          // block we will clean this tech debt together in the following PRs
          if (iter->isTensor()) {
            tracer::addInputs(node, args[i].name().c_str(), iter->toTensor());
          } else if (iter->isDouble()) {
            tracer::addInputs(node, args[i].name().c_str(), iter->toDouble());
          } else if (iter->isInt()) {
            tracer::addInputs(node, args[i].name().c_str(), iter->toInt());
          } else if (iter->isBool()) {
            tracer::addInputs(node, args[i].name().c_str(), iter->toBool());
          } else if (iter->isString()) {
            tracer::addInputs(
                node, args[i].name().c_str(), iter->toStringRef());
          } else if (iter->isDoubleList()) {
            tracer::addInputs(
                node, args[i].name().c_str(), iter->toDoubleList()->elements());
          } else if (iter->isIntList()) {
            tracer::addInputs(
                node, args[i].name().c_str(), iter->toIntList()->elements());
          } else if (iter->isBoolList()) {
            tracer::addInputs(
                node, args[i].name().c_str(), iter->toBoolList()->elements());
          } else if (iter->isTensorList()) {
            tracer::addInputs(
                node, args[i].name().c_str(), iter->toTensorList()->elements());
          } else if (iter->isNone()) {
            AT_ASSERT(args[i].type()->kind() == TypeKind::OptionalType);
            // create the none node with the right element type
            Value* none =
                graph
                    ->insertNode(graph->createNone(
                        reinterpret_cast<OptionalType*>(args[i].type().get())
                            ->getElementType()))
                    ->output();
            node->addInput(none);
          } else {
            throw std::runtime_error("unsupported input type.");
          }
        }
        graph->insertNode(node);
      }

      c10::Dispatcher::singleton().lookup(op, &stack).call(&stack);

      // wrap tensor outputs as variable
      for (auto iter = stack.end() - output_size; iter != stack.end(); ++iter) {
        if (iter->isTensor()) {
          *iter = torch::autograd::make_variable(std::move(*iter).toTensor());
        }
      }

      if (jit::tracer::isTracing()) {
        int i = 0;
        for (auto iter = stack.end() - output_size; iter != stack.end();
             ++iter, ++i) {
          if (iter->isTensor()) {
            tracer::addOutput(node, iter->toTensor());
          } else if (iter->isTensorList()) {
            tracer::addOutput(node, iter->toTensorList()->elements());
          } else {
            throw std::runtime_error("unsupported output type.");
          }
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
