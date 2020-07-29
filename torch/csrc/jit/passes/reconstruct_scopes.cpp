#include <torch/csrc/jit/passes/reconstruct_scopes.h>
#include <torch/csrc/jit/jit_log.h>

namespace torch {
namespace jit {

class ReconstructScopesPass {
 public:
  ReconstructScopesPass(const Module& m, Graph& g, std::string p)
      : root_module(m),
        graph(g),
        prefix(std::move(p)),
        has_duplicated_function(false){};
  void run();

 private:
  const Module& root_module;
  Graph& graph;
  std::string prefix;
  bool has_duplicated_function;

  std::unordered_map<Function*, ModulePtr> func_to_module;
  std::unordered_map<ModulePtr, std::string> module_names;

  void visitBlock(Block* b, const std::string& root_scope_string);
  void visitNode(Node* n, const std::string& root_scope_string);

  std::string getSubModuleName(const Module& module, const std::string& prefix);
  void constructFunctionToModuleMap(const Module& module);
  void constructRelativeNamesForModules(
      const Module& module,
      const std::string& prefix);

  std::string getScopeString(const InlinedCallStackEntry& frame) const;
};

void ReconstructScopesPass::constructFunctionToModuleMap(const Module& module) {
  for (const auto& method : module.get_methods()) {
    Function* func_ptr = &method.function();
    if (!has_duplicated_function &&
        func_to_module.find(func_ptr) != func_to_module.end()) {
      has_duplicated_function = true;
    }
    func_to_module[func_ptr] = module._ivalue();
  }
  for (const Module& m : module.children()) {
    constructFunctionToModuleMap(m);
  }
}

std::string ReconstructScopesPass::getSubModuleName(
    const Module& module,
    const std::string& prefix) {
  std::string moduleType = module.type()->str();
  size_t lastDotIndex = moduleType.rfind('.');
  if (lastDotIndex != std::string::npos) {
    moduleType = moduleType.substr(lastDotIndex + 1);
  }
  return prefix + "(" + moduleType + ")";
}

void ReconstructScopesPass::constructRelativeNamesForModules(
    const Module& module,
    const std::string& prefix) {
  module_names[module._ivalue()] = getSubModuleName(module, prefix);
  for (const NameModule& s : module.named_children()) {
    std::string newPrefix = (!has_duplicated_function)
        ? module_names[module._ivalue()] + "." + s.name
        : module_names[module._ivalue()] + ".";
    constructRelativeNamesForModules(s.value, newPrefix);
  }
}

std::string ReconstructScopesPass::getScopeString(
    const InlinedCallStackEntry& frame) const {
  Function* f = frame.first;
  if (!func_to_module.count(f)) {
    return "<null (no func in the map)>";
  }
  auto m = func_to_module.at(f);
  if (!module_names.count(m)) {
    return "<null (no module in the map)>";
  }
  std::string scopeString = module_names.at(m) + "." + f->name();

  if (has_duplicated_function) {
    SourceRange r = frame.second;
    if (r.source()) {
      if (auto orig = r.source()->findSourceRangeThatGenerated(r)) {
        r = *orig;
      }
    }
    if (auto file_line_col = r.file_line_col()) {
      std::string filename;
      size_t line, col;
      std::tie(filename, line, col) = *file_line_col;
      scopeString += "(" + filename + ":" + c10::to_string(line) + ":" +
          c10::to_string(col) + ")";
    }
  }
  return scopeString;
}

void ReconstructScopesPass::visitNode(
    Node* n,
    const std::string& root_scope_string) {
  for (Block* b : n->blocks()) {
    visitBlock(b, root_scope_string);
  }
  ScopePtr sc = c10::make_intrusive<Scope>();
  if (!n->callstack()) {
    sc = sc->push(Symbol::scope(root_scope_string));
  } else {
    for (const auto& frame : (*n->callstack())->vec()) {
      auto name = getScopeString(frame);
      GRAPH_UPDATE("Adding a scope ", name, " for node ", *n);
      sc = sc->push(Symbol::scope(name));
    }
  }
  n->setScope(sc);
  GRAPH_UPDATE("Updated node: ", *n);
}

void ReconstructScopesPass::visitBlock(
    Block* b,
    const std::string& root_scope_string) {
  for (Node* n : b->nodes()) {
    visitNode(n, root_scope_string);
  }
}

void ReconstructScopesPass::run() {
  GRAPH_DUMP("Graph before reconstructing scope", &graph);
  func_to_module.clear();
  module_names.clear();

  constructFunctionToModuleMap(root_module);
  constructRelativeNamesForModules(root_module, prefix);

  if (has_duplicated_function) {
    TORCH_WARN(
        "There are some duplicated module instances.\n",
        "Displaying only the class names of submodules.");
  }

  std::string root_scope_string =
      getSubModuleName(root_module, prefix) + ".forward";
  visitBlock(graph.block(), root_scope_string);
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
