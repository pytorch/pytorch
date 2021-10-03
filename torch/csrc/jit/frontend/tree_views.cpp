#include <torch/csrc/jit/frontend/tree_views.h>

namespace torch {
namespace jit {

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

} // namespace jit
} // namespace torch
