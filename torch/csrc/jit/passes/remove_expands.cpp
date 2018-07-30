#include "torch/csrc/jit/passes/remove_expands.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"

namespace torch { namespace jit {

static void RemoveExpands(Block* block) {
  for (auto it = block->nodes().begin(), end = block->nodes().end(); it != end;
       ++it) {
    for (auto sub : it->blocks())
      RemoveExpands(sub);
    if (it->kind() == aten::expand && it->get<int64_t>(attr::implicit) != static_cast<int64_t>(0)) {
      it->output()->replaceAllUsesWith(it->namedInput(attr::self));
      it.destroyCurrent();
    }
  }
}

void RemoveExpands(const std::shared_ptr<Graph>& graph) {
  RemoveExpands(graph->block());
}


}}
