#include <torch/csrc/jit/passes/refine_types.h>

#include <ATen/core/type_factory.h>

namespace torch {
namespace jit {

namespace {
static void VisitTupleNode(Node* node) {
  TORCH_CHECK(
      node->outputs().size() == 1, "Tuple must have exactly one output!");

  Value* output = node->outputs()[0];
  TupleType* tuple_type = dynamic_cast<TupleType*>(output->type().get());

  TORCH_CHECK(tuple_type, "Existing result type must be a Tuple!");
  TORCH_CHECK(
      tuple_type->containedTypes().size() == node->inputs().size(),
      "Number of contained types does not match number of inputs!");

  // Extract updated types from input values.
  std::vector<c10::TypePtr> types;
  for (const Value* input : node->inputs()) {
    types.push_back(input->type());
  }

  // Construct new tuple type based on input types.
  if (tuple_type->names()) {
    output->setType(c10::TupleType::createNamed(
        tuple_type->name(), tuple_type->names().value(), types));
  } else {
    output->setType(c10::TupleType::create(types));
  }
}

static void VisitNode(Node* node) {
  if (node->kind() == prim::TupleConstruct) {
    VisitTupleNode(node);
  }
}

static void RefineTypes(Block* block) {
  for (const auto& n : block->nodes()) {
    VisitNode(n);
  }
}
} // anonymous namespace

void RefineTypes(const std::shared_ptr<Graph>& graph) {
  RefineTypes(graph->block());
}

} // namespace jit
} // namespace torch
