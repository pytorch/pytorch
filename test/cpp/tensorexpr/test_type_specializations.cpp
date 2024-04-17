#include <gtest/gtest.h>

#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/jit/frontend/ir_emitter.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/pass_manager.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>

// Test that tensor type specializations are available in
// the custom passes

namespace torch {
namespace jit {

namespace {

bool hasTensorTypeSpecializations(torch::jit::Block* block) {
  for (Value* v : block->inputs()) {
    if (hasTensorTypeSpecialization(v))
      return true;
  }
  for (Node* n : block->nodes()) {
    for (torch::jit::Block* b : n->blocks()) {
      if (hasTensorTypeSpecializations(b))
        return true;
    }
    for (Value* v : n->outputs()) {
      if (hasTensorTypeSpecialization(v))
        return true;
    }
  }
  return false;
}

static bool hasSpecializations = false;
void detectTTSpecializationPass(std::shared_ptr<Graph>& graph) {
  GRAPH_DUMP("In detectTTSpecialization Custom Post Pass: ", graph);
  hasSpecializations = hasTensorTypeSpecializations(graph->block());
}

} // namespace

TEST(SpecializationsInCustomPasses, Basic) {
  RegisterPass p(detectTTSpecializationPass);
  hasSpecializations = false;
  std::shared_ptr<Graph> graph = std::make_shared<Graph>();
  parseIR(
      R"IR(
graph(%a.1 : Tensor,
      %b.1 : Tensor):
  %c.1 : Tensor = aten::mul(%a.1, %b.1) # misc/test_specializations.py:5:8
  %d.1 : Tensor = aten::mul(%c.1, %b.1) # misc/test_specializations.py:6:8
  return (%d.1)
  )IR",
      &*graph);

  IValue ival = IValue(torch::randn({22}, at::kCPU));
  std::vector<IValue> stack = {ival, ival};
  auto run = [&](std::shared_ptr<Graph>& graph, std::vector<IValue> stack) {
    GraphExecutor executor(graph, "");
    executor.run(stack);
    return stack;
  };
  run(graph, stack);

  // Profiling mode will not be run with simple executor
  if (!getExecutorMode()) {
    EXPECT_TRUE(hasSpecializations);
  }
}

} // namespace jit
} // namespace torch
