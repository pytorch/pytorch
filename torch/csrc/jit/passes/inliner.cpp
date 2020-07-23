#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/frontend/error_report.h>
#include <torch/csrc/jit/jit_log.h>

namespace torch {
namespace jit {

namespace prim {
using namespace ::c10::prim;
}

namespace {
void pushScopeInfo(Node* node) {
  auto callstack_ptr = *(node->callstack());
  const auto& vec = callstack_ptr->vec_with_module_info();
  ScopePtr sc = c10::make_intrusive<Scope>();
  for (const auto& tup : vec) {
    const auto opt_module_instance_info = std::get<2>(tup);
    if (opt_module_instance_info) {
      const auto& module_instance_info = opt_module_instance_info.value();
      if(module_instance_info.class_type()) {
        const auto& class_type = module_instance_info.class_type();
        const auto& instance_name = module_instance_info.instance_name();
        auto type_name = class_type->name()->qualifiedName();
        type_name = type_name.substr(type_name.find_last_of(".")+1);
        std::string module_fn_name = type_name + "(" + instance_name + ")::";
        module_fn_name += std::get<0>(tup)->name();
        sc = sc->push(Symbol::scope(module_fn_name));
      } else {
        sc = sc->push(Symbol::scope("TYPE_INFO_UNKNOWN::"));
      }
    }
    else {
      std::string free_fn = "FreeFunction::" + std::get<0>(tup)->name();
      sc = sc->push(Symbol::scope(free_fn));
    }
  }
  node->setScope(sc);
}

void ReconstructScopeFromInlinedCallStackHelper(Block* block) {
  for(auto node : block->nodes()) {
    if (node->callstack()) {
      pushScopeInfo(node);
    }
    for (auto b : node->blocks()) {
      ReconstructScopeFromInlinedCallStackHelper(b);
    }
  }
}
} // namespace

void ReconstructScopeFromInlinedCallStack(torch::jit::Graph& g) {
  ReconstructScopeFromInlinedCallStackHelper(g.block());
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
