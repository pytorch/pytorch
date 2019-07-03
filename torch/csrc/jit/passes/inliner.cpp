#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/script/error_report.h>
#include <torch/csrc/jit/script/module.h>

namespace torch {
namespace jit {

namespace prim {
using namespace ::c10::prim;
}

static void replace(
    Node* to_replace,
    const std::shared_ptr<Function>& fn,
    at::ArrayRef<Value*> inputs) {
  WithInsertPoint guard(to_replace);
  auto new_output =
      inlineCallTo(*to_replace->owningGraph(), *fn->graph(), inputs).at(0);
  if (to_replace->output()->hasDebugName()) {
    new_output->setDebugName(to_replace->output()->debugName());
  }
  to_replace->output()->replaceAllUsesWith(new_output);
}

void inlineCalls(Block* block) {
  for (auto it = block->nodes().begin(), end = block->nodes().end();
       it != end;) {
    Node* cur = *it++;
    switch (cur->kind()) {
      case prim::CallFunction: {
        AT_ASSERT(cur->inputs().at(0)->node()->kind() == prim::Constant);
        auto function_constant = cur->inputs().at(0)->node();
        auto fun_type =
            function_constant->output()->type()->expect<FunctionType>();
        replace(cur, fun_type->function(), cur->inputs().slice(1));
        cur->destroy();
        if (!function_constant->hasUses()) {
          function_constant->destroy();
        }
      } break;
      case prim::CallMethod: {
        const std::string& name = cur->s(attr::name);
        auto function =
            cur->inputs().at(0)->type()->expect<ClassType>()->getMethod(name);
        replace(cur, function, cur->inputs());
        cur->destroy();
      } break;
      default: {
        for (auto b : cur->blocks()) {
          inlineCalls(b);
        }
      } break;
    }
  }
}

void Inline(Graph& graph) {
  inlineCalls(graph.block());
}

} // namespace jit
} // namespace torch
