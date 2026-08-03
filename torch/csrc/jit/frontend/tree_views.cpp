#include <torch/csrc/jit/frontend/tree_views.h>

namespace torch::jit {

namespace {
void collectUnresolvedNames(
    std::vector<std::string>& names,
    const TreeView& node) {
  if (node.kind() == TK_ASSIGN) {
    for (const auto& expr : Assign{node.get()}.lhs_list()) {
      collectUnresolvedNames(names, expr);
    }
  } else if (node.kind() == TK_TUPLE_LITERAL) {
    for (const auto& expr : TupleLiteral{node.get()}.inputs()) {
      collectUnresolvedNames(names, expr);
    }
  } else if (node.kind() == TK_LIST_LITERAL) {
    for (const auto& expr : ListLiteral{node.get()}.inputs()) {
      collectUnresolvedNames(names, expr);
    }
  } else if (node.kind() == TK_VAR) {
    names.push_back(Var{node.get()}.name().name());
  }
}
} // namespace

std::vector<std::string> getUnresolvedClassAttributes(const ClassDef& def) {
  if (!def.assigns().present()) {
    return {};
  }
  std::vector<std::string> ret;
  for (const auto& assign : def.assigns().get()) {
    collectUnresolvedNames(ret, assign);
  }
  return ret;
}

/* static */ ClassDef ClassDef::create(
    const SourceRange& range,
    const Ident& name,
    const Maybe<Expr>& superclass,
    const List<Stmt>& body,
    const List<Property>& properties,
    const List<Assign>& assigns) {
  return ClassDef(Compound::create(
      TK_CLASS_DEF,
      range,
      {name,
       superclass,
       body,
       Maybe<List<Property>>::create(range, properties),
       Maybe<List<Assign>>::create(range, assigns)}));
}

} // namespace torch::jit
