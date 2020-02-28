#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/frontend/error_report.h>
#include <torch/csrc/jit/api/module.h>

namespace torch {
namespace jit {

namespace prim {
using namespace ::c10::prim;
}

void inlineCalls(Block* block) {
  for (auto it = block->nodes().begin(), end = block->nodes().end();
       it != end;) {
    Node* cur = *it++;
    switch (cur->kind()) {
      case prim::CallFunction: {
        AT_ASSERT(cur->input(0)->node()->kind() == prim::Constant);
        auto function_constant = cur->input(0)->node();
        auto fun_type =
            function_constant->output()->type()->expect<FunctionType>();
        auto fn_impl = dynamic_cast<FunctionImpl*>(fun_type->function());
        cur->removeInput(0);
        GRAPH_UPDATE("Inlining function '", fn_impl->name(), "' to ", *cur);
        GRAPH_UPDATE("Function body: ", *fn_impl->optimized_graph());
        inlineCallTo(cur, fn_impl);
      } break;
      case prim::CallMethod: {
        const std::string& name = cur->s(attr::name);
        if (auto class_type = cur->input(0)->type()->cast<ClassType>()) {
          auto function = class_type->getMethod(name);
          auto function_impl = dynamic_cast<FunctionImpl*>(function);
          GRAPH_UPDATE(
              "Inlining method '", function_impl->name(), "' to ", *cur);
          GRAPH_UPDATE("Function body: ", *function_impl->optimized_graph());
          inlineCallTo(cur, function);
        }
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
  GRAPH_DUMP("Before Inlining: ", &graph);
  inlineCalls(graph.block());
  GRAPH_DUMP("After Inlining: ", &graph);
}

} // namespace jit
} // namespace torch
