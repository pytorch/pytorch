#include <torch/csrc/jit/passes/reconstruct_scopes.h>
#include <torch/csrc/jit/jit_log.h>

namespace torch {
namespace jit {

void constructFunctionToModuleMap(
    script::Module& module,
    std::unordered_map<Function*, script::ModulePtr>& func_to_module) {
  for (auto& method : module.get_methods()) {
    func_to_module[&method.function()] = module.module_object();
  }
  for (script::Module m : module.children()) {
    constructFunctionToModuleMap(m, func_to_module);
  }
}

void constructRelativeNamesForModules(
    script::Module& module,
    std::unordered_map<script::ModulePtr, std::string>& module_names,
    const std::string& prefix) {
  module_names[module.module_object()] = prefix;
  for (script::NameModule s : module.named_children()) {
    constructRelativeNamesForModules(
        s.value, module_names, prefix + "." + s.name);
  }
}

void ReconstructScopesInBlock(
    const script::Module& module,
    Block& b,
    std::unordered_map<Function*, script::ModulePtr>& func_to_module,
    std::unordered_map<script::ModulePtr, std::string>& module_names,
    const std::string& prefix);

std::string getScopeString(
    Function* f,
    std::unordered_map<Function*, script::ModulePtr>& func_to_module,
    std::unordered_map<script::ModulePtr, std::string>& module_names) {
  if (!func_to_module.count(f)) {
    return "<null (no func in the map)>";
  }
  auto m = func_to_module.at(f);
  if (!module_names.count(m)) {
    return "<null (no module in the map)>";
  }
  return module_names.at(m) + "." + f->name();
}

void ReconstructScopeForNode(
    const script::Module& module,
    Node* n,
    std::unordered_map<Function*, script::ModulePtr>& func_to_module,
    std::unordered_map<script::ModulePtr, std::string>& module_names,
    const std::string& prefix = "") {
  for (auto b : n->blocks()) {
    ReconstructScopesInBlock(module, *b, func_to_module, module_names, "");
  }
  if (!n->callstack()) {
    return;
  }
  ScopePtr sc = c10::make_intrusive<Scope>();
  for (auto frame : (*n->callstack())->vec()) {
    auto name = getScopeString(frame.first, func_to_module, module_names);
    GRAPH_UPDATE("Adding a scope ", name, " for node ", *n);
    sc = sc->push(Symbol::scope(name));
  }
  n->setScope(sc);
  GRAPH_UPDATE("Updated node: ", *n);
}

void ReconstructScopesInBlock(
    const script::Module& module,
    Block& b,
    std::unordered_map<Function*, script::ModulePtr>& func_to_module,
    std::unordered_map<script::ModulePtr, std::string>& module_names,
    const std::string& prefix = "") {
  for (auto n : b.nodes()) {
    if (!n->callstack()) {
      continue;
    }
    ReconstructScopeForNode(module, n, func_to_module, module_names);
  }
}

void ReconstructScopesInGraph(
    script::Module& module,
    Graph& g,
    std::unordered_map<Function*, script::ModulePtr>& func_to_module,
    std::unordered_map<script::ModulePtr, std::string>& module_names,
    const std::string& prefix) {
  ReconstructScopesInBlock(module, *g.block(), func_to_module, module_names);
}

void ReconstructScopes(
    script::Module& module,
    Graph& g,
    const std::string& prefix = "top") {
  GRAPH_DUMP("Graph before reconstructing scope", &g);
  std::unordered_map<Function*, script::ModulePtr> func_to_module;
  constructFunctionToModuleMap(module, func_to_module);

  std::unordered_map<script::ModulePtr, std::string> module_names;
  constructRelativeNamesForModules(module, module_names, prefix);

  ReconstructScopesInGraph(module, g, func_to_module, module_names, prefix);
  GRAPH_DUMP("Graph after reconstructing scope", &g);
}

} // namespace jit
} // namespace torch
