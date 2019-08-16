#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/csrc/autograd/record_function.h>
#include <torch/csrc/jit/operator.h>
#include <torch/csrc/jit/tracer.h>

namespace torch {
namespace jit {
namespace {

at::Tensor wrap_tensor(at::Tensor&& tensor) {
  if (tensor.is_variable()) {
    return std::move(tensor);
  } else {
    return torch::autograd::make_variable(std::move(tensor));
  }
}

IValue wrap(IValue&& ivalue) {
  if (ivalue.isTensor()) {
    return wrap_tensor(std::move(ivalue).toTensor());
  } else if (ivalue.isTensorList()) {
    c10::List<at::Tensor> list = std::move(ivalue).toTensorList();
    for (size_t i = 0; i < list.size(); ++i) {
      list[i] = wrap_tensor(list.extract(i));
    }
    return std::move(list);
  } else if (ivalue.isGenericList()) {
    c10::impl::GenericList list = std::move(ivalue).toGenericList();
    for (size_t i = 0; i < list.size(); ++i) {
      list[i] = wrap(list.extract(i));
    }
    return std::move(list);
  } else if (ivalue.isGenericDict()) {
    for (auto& item : ivalue.toGenericDict()) {
      item.setValue(wrap(item.value()));
    }
    return std::move(ivalue);
  } else {
    return std::move(ivalue);
  }
}

// TODO This currently only handles tensors with requires_grad==False correctly.
//      It should also handle autograd.
Operator createOperatorFromC10(const c10::OperatorHandle& op) {
  return Operator(op, [op](Stack& stack) {
      RECORD_FUNCTION(op.schema().name(), stack);

      const auto input_size = op.schema().arguments().size();
      const auto output_size = op.schema().returns().size();

      Node* node = nullptr;

      // trace the input before unwrapping, otherwise we may lose
      // the input information
      if (jit::tracer::isTracing()) {
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
              Value* none =
                  graph
                      ->insertNode(graph->createNone(
                          reinterpret_cast<OptionalType*>(args[i].type().get())
                              ->getElementType()))
                      ->output();
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
      }

      c10::Dispatcher::singleton().lookup(op, &stack).call(&stack);

      // wrap tensor outputs as variable
      for (auto iter = stack.end() - output_size; iter != stack.end(); ++iter) {
        *iter = wrap(std::move(*iter));
      }

      if (jit::tracer::isTracing()) {
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
