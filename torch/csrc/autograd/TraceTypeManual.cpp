#include <ATen/TracerMode.h>
#include <ATen/core/op_registration/op_registration.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/utils/memory.h>
#include <torch/library.h>

using namespace at;

namespace torch {
namespace TraceType {

namespace {

Tensor& copy_(Tensor& self, const Tensor& src, bool non_blocking) {
  jit::Value* output = nullptr;
  if (torch::jit::tracer::isTracing()) {
    const jit::tracer::TracingState& state = *jit::tracer::getTracingState();
    auto& graph = state.graph;
    if (state.force_outplace && self.storage().use_count() <= 1) {
      // if you have no views of self, then an in place copy is equivalent to
      // making sure we expand src to the same size as self
      jit::Node* node = graph->create(jit::aten::expand_as, /*num_outputs=*/1);
      jit::tracer::addInputs(node, "src", src);
      jit::tracer::addInputs(node, "self", self);
      graph->insertNode(node);
      output = node->output();
    } else {
      output = graph->insert(
          jit::aten::copy_,
          {jit::tracer::getValueTrace(self), jit::tracer::getValueTrace(src)});
      jit::tracer::recordSourceLocation(output->node());
    }
    jit::tracer::ensureUniqueIfOutOfPlaced(
        "copy_ (possibly due to an assignment)", self);
  }

  {
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    self.copy_(src, non_blocking);
  }

  if (torch::jit::tracer::isTracing()) {
    jit::tracer::setOutput(output, self);
  }
  return self;
}

const Tensor& resize_(
    const Tensor& self,
    IntArrayRef size,
    c10::optional<MemoryFormat> optional_memory_format) {
  if (torch::jit::tracer::isTracing()) {
    if (jit::tracer::ArgumentStash::hasIntArrayRef("size")) {
      jit::tracer::ArgumentStash::popIntArrayRef("size");
    }
    jit::tracer::warn("resize_", jit::tracer::WARN_RESIZE);
    jit::tracer::delValueTrace(self);
  }

  {
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    // NOLINTNEXTLINE(performance-move-const-arg)
    self.resize_(size, std::move(optional_memory_format));
  }
  return self;
}

const Tensor& resize_as_(
    const Tensor& self,
    const Tensor& the_template,
    c10::optional<MemoryFormat> optional_memory_format) {
  if (torch::jit::tracer::isTracing()) {
    jit::tracer::warn("resize_as_", jit::tracer::WARN_RESIZE);
    jit::tracer::delValueTrace(self);
  }

  {
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    // NOLINTNEXTLINE(performance-move-const-arg)
    self.resize_as_(the_template, std::move(optional_memory_format));
  }
  return self;
}

Tensor detach(const Tensor& self) {
  torch::jit::Node* node = nullptr;
  if (jit::tracer::isTracing()) {
    auto& graph = jit::tracer::getTracingState()->graph;
    node = graph->create(jit::aten::detach, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    graph->insertNode(node);
  }

  auto result = [&]() {
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    return self.detach();
  }();

  if (jit::tracer::isTracing()) {
    jit::tracer::addOutput(node, result);
  }
  return result;
}

Tensor& detach_(Tensor& self) {
  torch::jit::Node* node = nullptr;
  if (jit::tracer::isTracing()) {
    auto& graph = jit::tracer::getTracingState()->graph;
    node = graph->create(jit::aten::detach, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("detach_", self);
  }

  {
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    self.detach_();
  }

  if (jit::tracer::isTracing()) {
    jit::tracer::addOutput(node, self);
  }
  return self;
}

// Invariant:
// - Ops registered to DispatchKey::Tracer below must be included in
// `MANUAL_TRACER` in tools/autograd/gen_variable_type.py
TORCH_LIBRARY_IMPL(aten, Tracer, m) {
  m.impl("resize_", resize_);
  m.impl("resize_as_", resize_as_);
  m.impl("detach", TORCH_FN(detach));
  m.impl("detach_", detach_);
  m.impl("copy_", copy_);

  // Skip tracing for the following ops by registering fallthrough kernel
  // explicitly.
  m.impl("_backward", CppFunction::makeFallthrough());
  m.impl("set_data", CppFunction::makeFallthrough());
  m.impl("data", CppFunction::makeFallthrough());
  m.impl("is_leaf", CppFunction::makeFallthrough());
  m.impl("output_nr", CppFunction::makeFallthrough());
  m.impl("_version", CppFunction::makeFallthrough());
  m.impl("requires_grad_", CppFunction::makeFallthrough());
  m.impl("retain_grad", CppFunction::makeFallthrough());
  m.impl("_fw_primal", CppFunction::makeFallthrough());
  m.impl("_make_dual", CppFunction::makeFallthrough());
}

} // namespace

} // namespace TraceType
} // namespace torch

namespace torch {
namespace jit {
void general_trace_function(const c10::OperatorHandle& op, Stack* stack) {
  const auto input_size = op.schema().arguments().size();
  const auto output_size = op.schema().returns().size();

  Node* node = nullptr;
  std::shared_ptr<tracer::TracingState> tracer_state;

  // trace the input before unwrapping, otherwise we may lose
  // the input information
  if (tracer::isTracing()) {
    tracer_state = tracer::getTracingState();
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
      if (type->isSubtypeOf(*TensorType::get())) {
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
        tracer::addInputs(node, args[i].name().c_str(), iter->toStringView());
      } else if (type->kind() == TypeKind::NumberType) {
        tracer::addInputs(node, args[i].name().c_str(), iter->toScalar());
      } else if (type->kind() == TypeKind::ListType) {
        const auto& elem_type = type->expectRef<ListType>().getElementType();
        if (elem_type->isSubtypeOf(*TensorType::get())) {
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
          tracer::addInputs(node, args[i].name().c_str(), objects, class_type);
        } else if (elem_type->kind() == TypeKind::FloatType) {
          AT_ASSERT(iter->isDoubleList());
          // NB: now, tracer doesn't support tracing double list. We add
          // special handling here, since in our case, we assume that all the
          // doubles in the list are constants
          auto value = iter->toDoubleVector();
          std::vector<Value*> info(value.size());
          for (const auto value_index : c10::irange(value.size())) {
            info[value_index] = graph->insertConstant(value[value_index]);
            tracer::recordSourceLocation(info[value_index]->node());
          }
          node->addInput(
              graph->insertNode(graph->createList(FloatType::get(), info))
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

    tracer::setTracingState(nullptr);
  }

  op.callBoxed(stack);

  if (tracer_state) {
    tracer::setTracingState(std::move(tracer_state));
    int i = 0;
    for (auto iter = stack->end() - output_size; iter != stack->end();
         ++iter, ++i) {
      const auto& type = op.schema().returns()[i].type();
      if (type->isSubtypeOf(*TensorType::get())) {
        AT_ASSERT(iter->isTensor());
        tracer::addOutput(node, iter->toTensor());
      } else if (type->kind() == TypeKind::ListType) {
        const auto& elem_type = type->expectRef<ListType>().getElementType();
        if (elem_type->isSubtypeOf(*TensorType::get())) {
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
        throw std::runtime_error(
            "unsupported output type: " + type->str() +
            ", from operator: " + toString(op.operator_name()));
      }
    }
  }
}
TORCH_LIBRARY_IMPL(_, Tracer, m) {
  m.fallback(CppFunction::makeFromBoxedFunction<&general_trace_function>());
}

} // namespace jit
} // namespace torch
