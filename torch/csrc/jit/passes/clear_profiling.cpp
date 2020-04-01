
#include <torch/csrc/jit/passes/clear_profiling.h>
#include <torch/csrc/jit/jit_log.h>

namespace torch {
namespace jit {

static void unprofileGraphInputs(const std::shared_ptr<Graph>& graph) {
  for (auto i : graph->inputs()) {
    if (i->type()->isSubtypeOf(TensorType::get())) {
      i->setType(unshapedType(i->type()));
    }
  }
}

static void unprofileBlock(Block* start_block) {
  std::vector<Block*> stack;
  stack.push_back(start_block);

  while (!stack.empty()) {
    Block* block = stack.back();
    stack.pop_back();

    for (auto n : block->nodes()) {
      for (auto o : n->outputs()) {
        if (o->type()->isSubtypeOf(TensorType::get())) {
          o->setType(unshapedType(o->type()));
        }
      }
      stack.insert(stack.end(), n->blocks().begin(), n->blocks().end());
    }
  }
}

void ClearProfilingInformation(const std::shared_ptr<Graph>& graph) {
  unprofileGraphInputs(graph);
  unprofileBlock(graph->block());
  GRAPH_DUMP("After ClearProfilingInformation: ", graph);
}

} // namespace jit
} // namespace torch
