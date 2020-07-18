#include <torch/csrc/jit/passes/reconstruct_scopes.h>
#include <torch/csrc/jit/jit_log.h>

namespace torch {
namespace jit {

class ReconstructScopesPass {
 public:
  ReconstructScopesPass(script::Module& m, Graph& g, const std::string& p)
      : root_module(&m), graph(&g), prefix(p){};
  void run();

 private:
  script::Module* root_module;
  Graph* graph;
  std::string prefix;

  std::unordered_map<Function*, script::ModulePtr> func_to_module;
  std::unordered_map<script::ModulePtr, std::string> module_names;

  void visitBlock(Block* b);
  void visitNode(Node* n);

  void constructFunctionToModuleMap(script::Module& module);
  void constructRelativeNamesForModules(
      script::Module& module,
      const std::string& prefix);

  std::string getScopeString(Function* f) const;
};

void ReconstructScopesPass::constructFunctionToModuleMap(
    script::Module& module) {
  for (auto& method : module.get_methods()) {
    func_to_module[&method.function()] = module._ivalue();
  }
  for (script::Module m : module.children()) {
    constructFunctionToModuleMap(m);
  }
}

void ReconstructScopesPass::constructRelativeNamesForModules(
    script::Module& module,
    const std::string& prefix) {
  module_names[module._ivalue()] = prefix;
  for (script::NameModule s : module.named_children()) {
    constructRelativeNamesForModules(s.value, prefix + "." + s.name);
  }
}

std::string ReconstructScopesPass::getScopeString(Function* f) const {
  if (!func_to_module.count(f)) {
    return "<null (no func in the map)>";
  }
  auto m = func_to_module.at(f);
  if (!module_names.count(m)) {
    return "<null (no module in the map)>";
  }
  return module_names.at(m) + "." + f->name();
}

void ReconstructScopesPass::visitNode(Node* n) {
  for (auto b : n->blocks()) {
    visitBlock(b);
  }
  if (!n->callstack()) {
    return;
  }
  ScopePtr sc = c10::make_intrusive<Scope>();
  for (auto frame : (*n->callstack())->vec()) {
    auto name = getScopeString(frame.first);
    GRAPH_UPDATE("Adding a scope ", name, " for node ", *n);
    sc = sc->push(Symbol::scope(name));
  }
  n->setScope(sc);
  GRAPH_UPDATE("Updated node: ", *n);
}

void ReconstructScopesPass::visitBlock(Block* b) {
  for (auto n : b->nodes()) {
    visitNode(n);
  }
}

void ReconstructScopesPass::run() {
  GRAPH_DUMP("Graph before reconstructing scope", graph);
  func_to_module.clear();
  module_names.clear();

  constructFunctionToModuleMap(*root_module);
  constructRelativeNamesForModules(*root_module, prefix);

  visitBlock(graph->block());
  GRAPH_DUMP("Graph after reconstructing scope", graph);
}

void ReconstructScopes(
    script::Module& module,
    Graph& g,
    const std::string& prefix = "top") {
  ReconstructScopesPass p(module, g, prefix);
  p.run();
}

} // namespace jit
} // namespace torch
