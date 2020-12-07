#include <torch/csrc/jit/passes/reconstruct_scopes.h>
#include <torch/csrc/jit/jit_log.h>

namespace torch {
namespace jit {

class ReconstructScopesPass {
 public:
  ReconstructScopesPass(const Module& m, Graph& g, std::string p)
      : root_module_(m),
        graph_(g),
        prefix_(std::move(p)),
        class_types_are_not_unique_(false){};
  void run();

 private:
  const Module& root_module_;
  Graph& graph_;
  std::string prefix_;

  // This boolean indicates whether there are two submodules of the same
  // class type. This issue may occur in a scripted module and make it
  // difficult to exactly track module information corresponding to each
  // Node* after inlining the graph. Consider the following example:

  // class A(nn.Module):
  //   def __init__(self):
  //     super(A, self).__init__()

  //   def forward(self, x):
  //     return x + 1

  // class B(nn.Module):
  //   def __init__(self):
  //     super(B, self).__init__()
  //     self.A0 = A()
  //     self.A1 = A()

  //   def forward(self, x):
  //     return self.A0(x) + self.A1(x)

  // m_traced = torch.jit.trace(B(), torch.Tensor([1]))
  // m_scripted = torch.jit.script(B())

  // In m_traced, self.A0 and self.A1 have different class types, but in
  // m_scripted, self.A0 and self.A1 have the same class types. Therefore,
  // it is difficult to distinguish 'A0' and 'A1' in the module hierarchy
  // after the graph is inlined. In this case, we add a warning to let
  // users know that the debugging information may be incomplete.
  bool class_types_are_not_unique_;

  std::unordered_map<Function*, ModulePtr> func_to_module_;
  std::unordered_map<ModulePtr, std::string> module_names_;

  void visitBlock(Block* b, const std::string& root_scope_string);
  void visitNode(Node* n, const std::string& root_scope_string);

  std::string getModuleTypeName(
      const Module& module,
      const std::string& prefix);
  void constructFunctionToModuleMap(const Module& module);
  void constructRelativeNamesForModules(
      const Module& module,
      const std::string& prefix);

  std::string getScopeString(const InlinedCallStackEntry& frame) const;

  void appendSourceRangeInfo(
      std::string& scopeString,
      const InlinedCallStackEntry& frame) const;
};

void ReconstructScopesPass::constructFunctionToModuleMap(const Module& module) {
  for (const auto& method : module.get_methods()) {
    Function* func_ptr = &method.function();
    if (!class_types_are_not_unique_ &&
        func_to_module_.find(func_ptr) != func_to_module_.end()) {
      class_types_are_not_unique_ = true;
    }
    func_to_module_[func_ptr] = module._ivalue();
  }
  for (const Module& m : module.children()) {
    constructFunctionToModuleMap(m);
  }
}

std::string ReconstructScopesPass::getModuleTypeName(
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
  module_names_[module._ivalue()] = getModuleTypeName(module, prefix);
  for (const NameModule& s : module.named_children()) {
    constructRelativeNamesForModules(
        s.value, module_names_[module._ivalue()] + "." + s.name);
  }
}

void ReconstructScopesPass::appendSourceRangeInfo(
    std::string& scopeString,
    const InlinedCallStackEntry& frame) const {
  SourceRange r = std::get<1>(frame);
  if (r.source()) {
    if (auto orig = r.source()->findSourceRangeThatGenerated(r)) {
      r = *orig;
    }
  }
  if (auto file_line_col = r.file_line_col()) {
    std::string filename;
    size_t line, col;
    std::tie(filename, line, col) = *file_line_col;
    scopeString += "<" + filename + ":" + c10::to_string(line) + ":" +
        c10::to_string(col) + ">";
  }
}

std::string ReconstructScopesPass::getScopeString(
    const InlinedCallStackEntry& frame) const {
  Function* f = std::get<0>(frame);
  if (!func_to_module_.count(f)) {
    return "<null (no func in the map)>";
  }
  auto m = func_to_module_.at(f);
  if (!module_names_.count(m)) {
    return "<null (no module in the map)>";
  }
  std::string scopeString = module_names_.at(m) + "." + f->name();

  // When class types are not unique, the module information may be
  // incomplele. In this case, we add source range information,
  // which can be helpful for debugging purposes.
  if (class_types_are_not_unique_) {
    appendSourceRangeInfo(scopeString, frame);
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
  GRAPH_DUMP("Graph before reconstructing scope", &graph_);
  func_to_module_.clear();
  module_names_.clear();

  constructFunctionToModuleMap(root_module_);
  constructRelativeNamesForModules(root_module_, prefix_);

  if (class_types_are_not_unique_) {
    TORCH_WARN(
        "It seems that the module contain two instances of the same class type.\n",
        "The current debugging program has not provided support for distinguishing ",
        "the two instances of the same class type.\n",
        "The module debugging information may be incomplete.");
  }

  std::string root_scope_string =
      getModuleTypeName(root_module_, prefix_) + ".forward";
  visitBlock(graph_.block(), root_scope_string);
  GRAPH_DUMP("Graph after reconstructing scope", &graph_);
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
