#pragma once

#include <optional>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <ATen/core/ivalue.h>
#include <nlohmann/json.hpp>

namespace torch::nativert {

class TreeSpec;

using TreeFlattenFn =
    void (*)(const c10::IValue&, const TreeSpec&, std::vector<c10::IValue>&);
using TreeUnflattenFn =
    c10::IValue (*)(std::vector<c10::IValue>, const nlohmann::json&);

using ContextLoadFn = nlohmann::json (*)(std::string_view);

using TreeMapFn = c10::function_ref<c10::IValue(const c10::IValue&)>;
using TreeMapNoReturnFn = c10::function_ref<void(const c10::IValue&)>;

using LeafApplyFn =
    void (*)(TreeMapNoReturnFn, const c10::IValue&, const TreeSpec&);

nlohmann::json defaultContextLoadFn(std::string_view);

struct NodeDef {
  TreeFlattenFn flattenFn;
  TreeUnflattenFn unflattenFn;
  LeafApplyFn leafApplyFn;

  ContextLoadFn contextLoadFn = defaultContextLoadFn;
};

class TreeSpec {
 public:
  // Leaf node.
  TreeSpec() : numLeaves_(1) {}

  // Non leaf node.
  TreeSpec(
      std::string_view uniformName,
      nlohmann::json context,
      std::vector<TreeSpec> children,
      NodeDef nodeDefCache)
      : uniformName_(uniformName),
        context_(std::move(context)),
        children_(std::move(children)),
        nodeDefCache_(nodeDefCache),
        numLeaves_(0) {
    for (auto& child : children_) {
      numLeaves_ += child.numLeaves();
      allLeaves_ &= child.isLeaf();
    }
  }

  bool isLeaf() const {
    return !uniformName_;
  }

  std::string_view uniformName() const {
    TORCH_CHECK(uniformName_);
    return uniformName_.value();
  }

  const nlohmann::json& context() const {
    return context_;
  }

  const auto& children() const {
    return children_;
  }

  const TreeSpec& children(size_t i) const {
    return children_[i];
  }

  const NodeDef& nodeDefCache() const {
    return nodeDefCache_;
  }

  size_t numLeaves() const {
    return numLeaves_;
  }

  bool allLeaves() const {
    return allLeaves_;
  }

  c10::TypePtr toAtenType() const;

 private:
  // Only non leaf nodes have names.
  // Examples of uniform name: "builtins.tuple", "builtins.dict".
  std::optional<std::string> uniformName_;
  nlohmann::json context_;
  std::vector<TreeSpec> children_;

  // Cached fields.
  NodeDef nodeDefCache_;
  size_t numLeaves_;
  bool allLeaves_ = true;
};

void registerPytreeNode(std::string_view typeName, NodeDef nodeDef);

// Serialized json tree spec should be dumped from treespec_dumps() in
// torch.utils._pytree directly .
TreeSpec treeSpecLoads(std::string_view json);

c10::IValue treeUnflatten(
    std::vector<c10::IValue> leaves,
    const TreeSpec& spec);

std::vector<c10::IValue> treeFlatten(
    const c10::IValue& tree,
    const TreeSpec& spec);

std::vector<c10::IValue> treeFlattenFromArgs(
    const std::vector<c10::IValue>& args,
    const std::unordered_map<std::string, c10::IValue>& kwargs,
    const TreeSpec& spec);

std::vector<at::Tensor> treeFlattenToTensorList(
    const c10::IValue& tree,
    const TreeSpec& spec);

c10::IValue treeMap(TreeMapFn f, const c10::IValue& tree, const TreeSpec& spec);

c10::IValue TORCH_API argsToIValue(
    const std::vector<c10::IValue>& args,
    const std::unordered_map<std::string, c10::IValue>& kwargs);

std::
    pair<std::vector<c10::IValue>, std::unordered_map<std::string, c10::IValue>>
    treeMapArgs(
        TreeMapFn f,
        const std::vector<c10::IValue>& args,
        const std::unordered_map<std::string, c10::IValue>& kwargs,
        const TreeSpec& spec);

void leafApply(
    TreeMapNoReturnFn f,
    const c10::IValue& tree,
    const TreeSpec& spec);

void leafApplyFromArgs(
    TreeMapNoReturnFn fn,
    const std::vector<c10::IValue>& args,
    const std::unordered_map<std::string, c10::IValue>& kwargs,
    const TreeSpec& spec);

} // namespace torch::nativert
