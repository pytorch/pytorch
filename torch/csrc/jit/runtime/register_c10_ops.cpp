#include <ATen/core/ATenOpList.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/record_function.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <unordered_set>

namespace torch {
namespace jit {

namespace {

// custom ops don't do tracing/autograd in VariableType yet, we need to handle
// tracing here.
// TODO This currently only handles tensors with requires_grad==False correctly.
//      It should also handle autograd.
Operator createOperatorFromC10_withTracingHandledHere(
    const c10::OperatorHandle& op) {
  return Operator(op, [op](Stack* stack) {
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
      for (auto iter = stack->end() - input_size; iter != stack->end();
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
            type = type->expectRef<OptionalType>().getElementType();
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
          tracer::addInputs(node, args[i].name().c_str(), iter->toStringRef());
        } else if (type->kind() == TypeKind::NumberType) {
          tracer::addInputs(node, args[i].name().c_str(), iter->toScalar());
        } else if (type->kind() == TypeKind::ListType) {
          const auto& elem_type = type->expectRef<ListType>().getElementType();
          if (elem_type->isSubtypeOf(TensorType::get())) {
            AT_ASSERT(iter->isTensorList());
            auto list = iter->toTensorVector();
            tracer::addInputs(node, args[i].name().c_str(), list);
          } else if (auto class_type = elem_type->cast<ClassType>()) {
            AT_ASSERT(iter->isList());
            auto list = iter->toList();
            std::vector<c10::intrusive_ptr<c10::ivalue::Object>> objects;
            for (IValue iv : list) {
              objects.emplace_back(std::move(iv).toObject());
            }
            tracer::addInputs(
                node, args[i].name().c_str(), objects, class_type);
          } else if (elem_type->kind() == TypeKind::FloatType) {
            AT_ASSERT(iter->isDoubleList());
            // NB: now, tracer doesn't support tracing double list. We add
            // special handling here, since in our case, we assume that all the
            // doubles in the list are constants
            auto value = iter->toDoubleVector();
            std::vector<Value*> info(value.size());
            for (size_t value_index = 0; value_index < value.size();
                 ++value_index) {
              info[value_index] = graph->insertConstant(value[value_index]);
              tracer::recordSourceLocation(info[value_index]->node());
            }
            node->addInput(
                graph
                    ->insertNode(graph->createList(jit::FloatType::get(), info))
                    ->output());
          } else if (elem_type->kind() == TypeKind::IntType) {
            AT_ASSERT(iter->isIntList());
            tracer::addInputs(
                node,
                args[i].name().c_str(),
                c10::IntArrayRef(iter->toIntVector()));
          } else if (elem_type->kind() == TypeKind::BoolType) {
            AT_ASSERT(iter->isBoolList());
            tracer::addInputs(
                node, args[i].name().c_str(), iter->toBoolList().vec());
          } else {
            throw std::runtime_error(
                "unsupported input list type: " + elem_type->str());
          }
        } else if (iter->isObject()) {
          tracer::addInputs(node, args[i].name().c_str(), iter->toObject());
        } else {
          throw std::runtime_error("unsupported input type: " + type->str());
        }
      }
      graph->insertNode(node);

      jit::tracer::setTracingState(nullptr);
    }

    op.callBoxed(stack);

    if (tracer_state) {
      jit::tracer::setTracingState(std::move(tracer_state));
      int i = 0;
      for (auto iter = stack->end() - output_size; iter != stack->end();
           ++iter, ++i) {
        const auto& type = op.schema().returns()[i].type();
        if (type->isSubtypeOf(TensorType::get())) {
          AT_ASSERT(iter->isTensor());
          tracer::addOutput(node, iter->toTensor());
        } else if (type->kind() == TypeKind::ListType) {
          const auto& elem_type = type->expectRef<ListType>().getElementType();
          if (elem_type->isSubtypeOf(TensorType::get())) {
            AT_ASSERT(iter->isTensorList());
            tracer::addOutput(node, iter->toTensorList());
          } else {
            throw std::runtime_error(
                "unsupported ouptut list type: " + elem_type->str());
          }
        } else if (type->kind() == TypeKind::ClassType) {
          AT_ASSERT(iter->isObject());
          tracer::addOutput(node, iter->toObject());
        } else {
          throw std::runtime_error("unsupported output type: " + type->str());
        }
      }
    }
  });
}

Operator createOperatorFromC10_withTracingNotHandledHere(
    const c10::OperatorHandle& op) {
  return Operator(op, [op](Stack* stack) { op.callBoxed(stack); });
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
    if (at::is_custom_op(op.schema().operator_name())) {
      // custom ops don't do tracing/autograd in VariableType yet, we need to
      // handle tracing here.
      torch::jit::registerOperator(
          createOperatorFromC10_withTracingHandledHere(op));
    } else {
      // Ops from native_functions.yaml do tracing/autograd in VariableType,
      // no need to handle it here
      torch::jit::registerOperator(
          createOperatorFromC10_withTracingNotHandledHere(op));
    }
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
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
Registerer& dummy = registerer();

} // namespace

void ensure_c10_registerer_defined() {
  registerer();
}

} // namespace jit
} // namespace torch
