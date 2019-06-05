#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/script/error_report.h>
#include <torch/csrc/jit/script/module.h>

namespace torch {
namespace jit {

namespace prim {
using namespace ::c10::prim;
}

void inlineCalls(Block* block) {
  Node* cur = block->nodes().front();
  Node* end = block->return_node();

  while (cur != end) {
    auto next = cur->next();
    for (auto b : cur->blocks()) {
      inlineCalls(b);
    }
    if (cur->kind() == prim::CallFunction) {
      AT_ASSERT(cur->inputs().at(0)->node()->kind() == prim::Constant);
      auto function_constant = cur->inputs().at(0)->node();
      auto fun_type =
          function_constant->output()->type()->expect<FunctionType>();
      auto graph = fun_type->function()->graph();

      auto old_output = cur->outputs();
      // slice function ptr value
      auto inputs = cur->inputs().slice(1);
      WithInsertPoint guard(next);
      auto new_output =
          inlineCallTo(*cur->owningGraph(), *graph.get(), inputs).at(0);
      if (old_output.at(0)->hasUniqueName()) {
        auto name = old_output.at(0)->uniqueName();
        new_output->setUniqueName(name);
      }

      old_output.at(0)->replaceAllUsesWith(new_output);
      next = cur->next();
      cur->destroy();
      if (!function_constant->hasUses()) {
        function_constant->destroy();
      }
    }
    cur = next;
  }
}

void Inline(Graph& graph) {
  inlineCalls(graph.block());
}

} // namespace jit
} // namespace torch
