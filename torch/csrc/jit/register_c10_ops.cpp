#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/OpsAlreadyMovedToC10.h>
#include <torch/csrc/autograd/record_function.h>
#include <torch/csrc/jit/operator.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/tracer.h>
#include <unordered_set>

namespace torch {
namespace jit {

namespace {

// TODO This currently only handles tensors with requires_grad==False correctly.
//      It should also handle autograd.
Operator createOperatorFromC10(const c10::OperatorHandle& op) {
  return Operator(op, [op](Stack& stack) {
      RECORD_FUNCTION(op.schema().name(), stack);
      const auto input_size = op.schema().arguments().size();
      const auto output_size = op.schema().returns().size();

      Node* node = nullptr;
      std::shared_ptr<jit::tracer::TracingState> tracer_state;

      // trace the input before unwrapping, otherwise we may lose
      // the input information
      if (jit::tracer::isTracing()) {
        tracer_state = jit::tracer::getTracingState();
        auto symbol = Symbol::fromQualString(op.schema().name());
        const auto& graph = tracer::getTracingState()->graph;
        node = graph->create(symbol, 0);
        tracer::recordSourceLocation(node);
        const auto& args = op.schema().arguments();
        int i = 0;
        for (auto iter = stack.end() - input_size; iter != stack.end();
             ++iter, ++i) {
          // TODO we need to refactor graph APIs (e.g., addInputs)
          // appropriately; after that, we can get rid of the giant if-else
          // block we will clean this tech debt together in the following PRs
          auto type = args[i].type();
          if (type->kind() == TypeKind::OptionalType) {
            if (iter->isNone()) {
              Value* none = graph->insertNode(graph->createNone())->output();
              node->addInput(none);
              continue;
            } else {
              type =
                  reinterpret_cast<OptionalType*>(type.get())->getElementType();
            }
          }
          if (type->isSubtypeOf(TensorType::get())) {
            AT_ASSERT(iter->isTensor());
            tracer::addInputs(node, args[i].name().c_str(), iter->toTensor());
          } else if (type->kind() == TypeKind::FloatType) {
            AT_ASSERT(iter->isDouble());
            tracer::addInputs(node, args[i].name().c_str(), iter->toDouble());
          } else if (type->kind() == TypeKind::IntType) {
            AT_ASSERT(iter->isInt());
            tracer::addInputs(node, args[i].name().c_str(), iter->toInt());
          } else if (type->kind() == TypeKind::BoolType) {
            AT_ASSERT(iter->isBool());
            tracer::addInputs(node, args[i].name().c_str(), iter->toBool());
          } else if (type->kind() == TypeKind::StringType) {
            AT_ASSERT(iter->isString());
            tracer::addInputs(
                node, args[i].name().c_str(), iter->toStringRef());
          } else if (type->kind() == TypeKind::ListType) {
            const auto& elem_type =
                reinterpret_cast<ListType*>(type.get())->getElementType();
            if (elem_type->isSubtypeOf(TensorType::get())) {
              AT_ASSERT(iter->isTensorList());
              auto list = iter->toTensorListRef();
              tracer::addInputs(node, args[i].name().c_str(), list);
            } else if (elem_type->kind() == TypeKind::FloatType) {
              AT_ASSERT(iter->isDoubleList());
              // NB: now, tracer doesn't support tracing double list. We add special
              // handling here, since in our case, we assume that all the doubles
              // in the list are constants
              auto value = iter->toDoubleListRef();
              std::vector<Value*> info(value.size());
              for (size_t value_index = 0; value_index < value.size(); ++value_index) {
                info[value_index] = graph->insertConstant(value[value_index]);
                tracer::recordSourceLocation(info[value_index]->node());
              }
              node->addInput(
                  graph->insertNode(graph->createList(jit::FloatType::get(), info))->output());
            } else if (elem_type->kind() == TypeKind::IntType) {
              AT_ASSERT(iter->isIntList());
              tracer::addInputs(
                  node, args[i].name().c_str(), iter->toIntListRef());
            } else if (elem_type->kind() == TypeKind::BoolType) {
              AT_ASSERT(iter->isBoolList());
              tracer::addInputs(
                  node, args[i].name().c_str(), c10::impl::toVector(iter->toBoolList()));
            } else {
              throw std::runtime_error(
                  "unsupported input list type: " + elem_type->str());
            }
          } else {
            throw std::runtime_error("unsupported input type: " + type->str());
          }
        }
        graph->insertNode(node);

        jit::tracer::setTracingState(nullptr);
      }

#ifdef USE_STATIC_DISPATCH
      {
        at::AutoNonVariableTypeMode non_var_type_mode(true);
        c10::Dispatcher::singleton().callBoxed(op, &stack);
      }
#else
      c10::Dispatcher::singleton().callBoxed(op, &stack);
#endif // USE_STATIC_DISPATCH

      if (tracer_state) {
        jit::tracer::setTracingState(std::move(tracer_state));
        int i = 0;
        for (auto iter = stack.end() - output_size; iter != stack.end();
             ++iter, ++i) {
          const auto& type = op.schema().returns()[i].type();
          if (type->isSubtypeOf(TensorType::get())) {
            AT_ASSERT(iter->isTensor());
            tracer::addOutput(node, iter->toTensor());
          } else if (type->kind() == TypeKind::ListType) {
            const auto& elem_type =
                reinterpret_cast<ListType*>(type.get())->getElementType();
            if (elem_type->isSubtypeOf(TensorType::get())) {
              AT_ASSERT(iter->isTensorList());
              tracer::addOutput(node, iter->toTensorList());
            } else {
              throw std::runtime_error(
                  "unsupported ouptut list type: " + elem_type->str());
            }
          } else {
            throw std::runtime_error("unsupported output type: " + type->str());
          }
        }
      }

      return 0;
  });
}

class RegistrationListener final : public c10::OpRegistrationListener {
public:
  void onOperatorRegistered(const c10::OperatorHandle& op) override {
    if(at::is_aten_op(op.schema().operator_name())) {
      // Ignore ATen ops for now because they have their own code
      // to expose them to JIT in register_aten_ops.cpp
      // TODO Remove register_aten_ops.cpp and also use this registration here
      return;
    }
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
