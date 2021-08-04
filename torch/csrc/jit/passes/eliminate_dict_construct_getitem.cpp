#include <torch/csrc/jit/passes/eliminate_dict_construct_getitem.h>

namespace torch {
namespace jit {

namespace {

std::unordered_map<Value*, Value*> GetKeyValuesFromDictNode(Node* dict) {
  TORCH_CHECK(
      dict->inputs().size() % 2 == 0,
      "DictConstruct should have an even number of inputs");
  std::unordered_map<Value*, Value*> key_values;

  for (size_t i = 0; i < dict->inputs().size(); i += 2) {
    auto* key = dict->input(i);
    auto* value = dict->input(i + 1);
    key_values.emplace(key, value);
  }

  return key_values;
}

void UpdateGetItemUses(Node* dict, const std::vector<Node*>& dict_users) {
  auto key_values = GetKeyValuesFromDictNode(dict);

  for (auto* user : dict_users) {
    auto* target_key = user->input(1);
    auto kv = key_values.find(target_key);
    if (kv != key_values.end()) {
      // Replace all uses of the getitem call with the value in the
      // DictConstruct call.
      user->output()->replaceAllUsesWith(kv->second);
    }
  }
}

bool IsPureOp(const NodeKind& kind) {
  return kind == aten::__getitem__ || kind == aten::len || kind == aten::copy ||
      kind == aten::keys || kind == aten::values || kind == aten::__contains__;
}

} // namespace

void EliminateDictConstructGetItem(std::shared_ptr<Graph>& graph) {
  for (Node* n : graph->nodes()) {
    bool dict_mutated = false;
    std::vector<Node*> users;
    if (n->kind() == prim::DictConstruct) {
      auto* dict = n->output();
      for (auto& use : dict->uses()) {
        auto* user = use.user;
        if (!IsPureOp(user->kind())) {
          dict_mutated = true;
          break;
        }

        if (user->kind() == aten::__getitem__) {
          users.push_back(user);
        }
      }
      if (!dict_mutated) {
        UpdateGetItemUses(n, users);
      }
    }
  }
}

} // namespace jit
} // namespace torch
