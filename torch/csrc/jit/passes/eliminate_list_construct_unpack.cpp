#include <torch/csrc/jit/passes/eliminate_list_construct_unpack.h>

namespace torch {
namespace jit {

namespace {

void UpdateUnpackUses(Node* list, const std::vector<Node*>& list_unpack_users) {
  size_t nInputs = list->inputs().size();
  for (auto* user : list_unpack_users) {
    TORCH_CHECK(
        nInputs == user->outputs().size(),
        "The number of ListConstruct inputs and the number of ListUnpack outputs should be same.")
    // Replace all uses of the ListUnpack outputs with the value in the
    // ListConstruct call.
    for (size_t i = 0; i < nInputs; ++i) {
      user->output(i)->replaceAllUsesWith(list->input(i));
    }
  }
}

bool IsPureOp(const NodeKind& kind) {
  return kind == prim::ListUnpack || kind == aten::__getitem__ ||
      kind == aten::len || kind == aten::copy;
}

} // namespace

void EliminateListConstructUnpack(std::shared_ptr<torch::jit::Graph>& graph) {
  for (Node* n : graph->nodes()) {
    bool list_mutated = false;
    std::vector<Node*> users;

    if (n->kind() != prim::ListConstruct) {
      continue;
    }
    auto* list = n->output();
    for (auto& use : list->uses()) {
      auto* user = use.user;
      if (!IsPureOp(user->kind())) {
        list_mutated = true;
        break;
      }

      if (user->kind() == prim::ListUnpack) {
        users.push_back(user);
      }
    }
    if (!list_mutated) {
      UpdateUnpackUses(n, users);
    }
  }
}

} // namespace jit
} // namespace torch
