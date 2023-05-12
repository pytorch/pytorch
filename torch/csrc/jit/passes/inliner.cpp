#include <torch/csrc/jit/passes/inliner.h>

#include <ATen/core/interned_strings.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/frontend/error_report.h>
#include <torch/csrc/jit/jit_log.h>

namespace torch {
namespace jit {

namespace prim {
using namespace ::c10::prim;
}

GraphFunction* tryToGraphFunction(Node* n) {
  if (n->kind() == prim::CallFunction) {
    AT_ASSERT(n->input(0)->node()->kind() == prim::Constant);
    auto function_constant = n->input(0)->node();
    auto fun_type = function_constant->output()->type()->expect<FunctionType>();
    return tryToGraphFunction(*fun_type->function());
  }
  if (n->kind() == prim::CallMethod) {
    const std::string& name = n->s(attr::name);
    if (auto class_type = n->input(0)->type()->cast<ClassType>()) {
      Function& function = class_type->getMethod(name);
      return tryToGraphFunction(function);
    }
  }
  return nullptr;
}

void inlineCalls(Block* block) {
  for (auto it = block->nodes().begin(), end = block->nodes().end();
       it != end;) {
    Node* cur = *it++;
    switch (cur->kind()) {
      case prim::CallFunction: {
        if (auto graphFunction = tryToGraphFunction(cur)) {
          auto function_constant = cur->input(0)->node();
          auto fun_type =
              function_constant->output()->type()->expect<FunctionType>();

          cur->removeInput(0);
          GRAPH_UPDATE(
              "Inlining function '",
              fun_type->function()->name(),
              "' to ",
              *cur);

          std::shared_ptr<Graph> g = nullptr;
          // inline optimized graph for debugging/testing purposes.
          // we only insert fallback functions in JIT optimized graphs for
          // execution, not on the Graph that is used for serialization
          bool fallback =
              function_constant->hasAttribute(Symbol::attr("fallback"));
          if (fallback && graphFunction->get_executor().isOptimized()) {
            auto exec_plans =
                graphFunction->get_executor().getDebugState().execution_plans;
            if (!exec_plans.empty()) {
              g = exec_plans.begin()->second.graph;
              // optimized_graph() calls Inline, so we only need to explicitly
              // invoke inlining on the jit optimized graph with recursive
              // fallback function calls
              Inline(*g.get());
            }
          }
          if (g == nullptr) {
            g = graphFunction->optimized_graph();
          }

          GRAPH_UPDATE("Function body: ", g);
          inlineCallTo(cur, graphFunction, g.get());
        }
      } break;
      case prim::CallMethod: {
        if (auto graphFunction = tryToGraphFunction(cur)) {
          GRAPH_UPDATE("Inlining method '", cur->s(attr::name), "' to ", *cur);
          GRAPH_UPDATE("Function body: ", graphFunction->optimized_graph());
          inlineCallTo(cur, graphFunction);
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
