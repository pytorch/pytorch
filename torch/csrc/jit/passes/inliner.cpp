#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/script/error_report.h>
#include <torch/csrc/jit/script/module.h>

namespace torch {
namespace jit {

namespace prim {
using namespace ::c10::prim;
}

void inlineCalls(Block* block, bool recurse) {
  for (auto it = block->nodes().begin(), end = block->nodes().end();
       it != end;) {
    // Explanation of what's going on with the iterator:
    //
    // Say the nodes are like:
    //   n1
    //   n2 <-- function call containing:
    //       n2_a
    //       n2_b
    //   n3
    //
    // Post-inlining, it looks like:
    //   n1
    //   n2_a
    //   n2_b
    //   n3
    //
    // Notably, n2 has been destroyed, which is why we have to do all these gymnastics.
    // If `recurse` is true:
    //   1. `it` starts pointing at n2
    //   2. Before inlining, we backtrack by one (n1)
    //   3. Inlining removes n2 and replaces it with n2_a and n2_b
    //   3. Advance the iterator, processing n2_a and all new nodes.
    // If `recurse` is false:
    //   1. Advance the iterator immediately, to point it to n3
    //   2. Inlining deletes n2
    //   3. We progress forward, skipping n2_a and n2_b

    Node* cur = recurse ? *it : *it++;
    switch (cur->kind()) {
      case prim::CallFunction: {
        AT_ASSERT(cur->input(0)->node()->kind() == prim::Constant);
        auto function_constant = cur->input(0)->node();
        auto fun_type =
            function_constant->output()->type()->expect<FunctionType>();
        cur->removeInput(0);
        if (recurse)  {
          it--;
        }
        inlineCallTo(cur, *fun_type->function()->graph());
      } break;
      case prim::CallMethod: {
        const std::string& name = cur->s(attr::name);
        auto function =
            cur->input(0)->type()->expect<ClassType>()->getMethod(name);
        if (recurse)  {
          it--;
        }
        inlineCallTo(cur, *function->graph());
      } break;
      default: {
        for (auto b : cur->blocks()) {
          inlineCalls(b, recurse);
        }
      } break;
    }
    if (recurse) {
      it++;
    }
  }
}

void Inline(Graph& graph, bool recurse) {
  inlineCalls(graph.block(), recurse);
}

} // namespace jit
} // namespace torch
