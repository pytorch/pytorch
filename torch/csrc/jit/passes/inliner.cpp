#include <torch/csrc/jit/passes/inliner.h>

#include <torch/csrc/jit/api/function_impl.h>
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

        if (auto graphFunction = tryToGraphFunction(*fun_type->function())) {
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
            if (exec_plans.size() != 0) {
              g = exec_plans.begin()->second.graph;
              // optimized_graph() calls Inline, so we only need to explicitly
              // invoke inlining on the jit optimized graph with recursive
              // fallback funciton calls
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
        const std::string& name = cur->s(attr::name);
        if (auto class_type = cur->input(0)->type()->cast<ClassType>()) {
          Function& function = class_type->getMethod(name);
          if (auto graphFunction = tryToGraphFunction(function)) {
            GRAPH_UPDATE("Inlining method '", function.name(), "' to ", *cur);
            GRAPH_UPDATE("Function body: ", graphFunction->optimized_graph());
            inlineCallTo(cur, graphFunction);
          }
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
