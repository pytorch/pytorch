#include <torch/csrc/jit/passes/clear_profiling.h>

#include <torch/csrc/jit/jit_log.h>

namespace torch::jit {

void unprofileGraphInputs(const std::shared_ptr<Graph>& graph) {
  for (auto i : graph->inputs()) {
    if (i->type()->isSubtypeOf(*TensorType::get())) {
      i->setType(unshapedType(i->type()));
    }
  }
}

void unprofileBlock(Block* start_block) {
  std::vector<Block*> stack;
  stack.push_back(start_block);

  while (!stack.empty()) {
    Block* block = stack.back();
    stack.pop_back();

    for (auto n : block->nodes()) {
      for (auto o : n->outputs()) {
        if (o->type()->isSubtypeOf(*TensorType::get())) {
          o->setType(unshapedType(o->type()));
        }
      }
      stack.insert(stack.end(), n->blocks().begin(), n->blocks().end());
    }
  }
}

// We need to make sure that passes that use profiling information
// use it **only after** guards validating it are inserted
// Ideally, we would run any pass that relies on profiling information
// after `InsertBailOuts`, however, practically, some passes
// (e.g. Peephole) useful to run both w/ and w/o profiling information
// so we could run them in `preoptimizeGraph` and
// in `runProfilingInsensitiveOptimizations`
void ClearProfilingInformation(const std::shared_ptr<Graph>& graph) {
  unprofileGraphInputs(graph);
  unprofileBlock(graph->block());
  GRAPH_DUMP("After ClearProfilingInformation: ", graph);
}

} // namespace torch::jit
