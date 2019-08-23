#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/script/error_report.h>
#include <torch/csrc/jit/script/module.h>

namespace torch {
namespace jit {

namespace prim {
using namespace ::c10::prim;
}

void inlineCalls(Block* block) {
  for (auto it = block->nodes().begin(), end = block->nodes().end();
       it != end;) {
    Node* cur = *it;
    switch (cur->kind()) {
      case prim::CallFunction: {
        AT_ASSERT(cur->input(0)->node()->kind() == prim::Constant);
        auto function_constant = cur->input(0)->node();
        auto fun_type =
            function_constant->output()->type()->expect<FunctionType>();
        cur->removeInput(0);
        // We need to decrement `it` because inlineCallTo will destroy `cur`
        // and potentially replace it with new nodes.
        it--;
        inlineCallTo(cur, *fun_type->function()->graph());
      } break;
      case prim::CallMethod: {
        const std::string& name = cur->s(attr::name);
        auto function =
            cur->input(0)->type()->expect<ClassType>()->getMethod(name);
        // We need to decrement `it` because inlineCallTo will destroy `cur`
        // and potentially replace it with new nodes.
        it--;
        inlineCallTo(cur, *function->graph());
      } break;
      default: {
        for (auto b : cur->blocks()) {
          inlineCalls(b);
        }
      } break;
    }
    it++;
  }
}

void Inline(Graph& graph) {
  inlineCalls(graph.block());
}

} // namespace jit
} // namespace torch
