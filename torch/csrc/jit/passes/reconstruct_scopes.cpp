#include <torch/csrc/jit/passes/reconstruct_scopes.h>
#include <torch/csrc/jit/jit_log.h>

namespace torch {
namespace jit {

class ReconstructScopesPass {
 public:
  ReconstructScopesPass(const Module& m, Graph& g, std::string p)
      : root_module(m), graph(g), prefix(std::move(p)){};
  void run();

 private:
  const Module& root_module;
  Graph& graph;
  std::string prefix;

  std::unordered_map<Function*, ModulePtr> func_to_module;
  std::unordered_map<ModulePtr, std::string> module_names;

  void visitBlock(Block* b);
  void visitNode(Node* n);

  void constructFunctionToModuleMap(const Module& module);
  void constructRelativeNamesForModules(
      const Module& module,
      const std::string& prefix);

  std::string getScopeString(Function* f) const;
};

void ReconstructScopesPass::constructFunctionToModuleMap(const Module& module) {
  for (const auto& method : module.get_methods()) {
    func_to_module[&method.function()] = module._ivalue();
  }
  for (const Module& m : module.children()) {
    constructFunctionToModuleMap(m);
  }
}

void ReconstructScopesPass::constructRelativeNamesForModules(
    const Module& module,
    const std::string& prefix) {
  module_names[module._ivalue()] = prefix;
  for (const NameModule& s : module.named_children()) {
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
  for (Block* b : n->blocks()) {
    visitBlock(b);
  }
  if (!n->callstack()) {
    return;
  }
  ScopePtr sc = c10::make_intrusive<Scope>();
  for (const auto& frame : (*n->callstack())->vec()) {
    auto name = getScopeString(frame.first);
    GRAPH_UPDATE("Adding a scope ", name, " for node ", *n);
    sc = sc->push(Symbol::scope(name));
  }
  n->setScope(sc);
  GRAPH_UPDATE("Updated node: ", *n);
}

void ReconstructScopesPass::visitBlock(Block* b) {
  for (Node* n : b->nodes()) {
    visitNode(n);
  }
}

void ReconstructScopesPass::run() {
  GRAPH_DUMP("Graph before reconstructing scope", &graph);
  func_to_module.clear();
  module_names.clear();

  constructFunctionToModuleMap(root_module);
  constructRelativeNamesForModules(root_module, prefix);

  visitBlock(graph.block());
  GRAPH_DUMP("Graph after reconstructing scope", &graph);
}

void ReconstructScopes(
    const Module& module,
    Graph& g,
    const std::string& prefix = "top") {
  ReconstructScopesPass p(module, g, prefix);
  p.run();
}

} // namespace jit
} // namespace torch
