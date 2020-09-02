#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/frontend/error_report.h>
#include <torch/csrc/jit/jit_log.h>

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
        if (!fun_type->function()->isGraphFunction()) {
          continue;
        }
        cur->removeInput(0);
        GRAPH_UPDATE(
            "Inlining function '", fun_type->function()->name(), "' to ", *cur);
        GRAPH_UPDATE(
            "Function body: ", *fun_type->function()->optimized_graph());
        inlineCallTo(cur, fun_type->function());
      } break;
      case prim::CallMethod: {
        const std::string& name = cur->s(attr::name);
        if (auto class_type = cur->input(0)->type()->cast<ClassType>()) {
          Function& function = class_type->getMethod(name);
          if (!function.isGraphFunction()) {
            continue;
          }
          GRAPH_UPDATE("Inlining method '", function.name(), "' to ", *cur);
          GRAPH_UPDATE("Function body: ", *function.optimized_graph());
          inlineCallTo(cur, &function);
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
