/*
 * A C++ extension bridge with the Python pytree
 * serialization/unserialization format for torch.export.
 */

#pragma once

#include <optional>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <ATen/core/ivalue.h>
#include <nlohmann/json.hpp>
#include <torch/nativert/graph/Graph.h>

namespace torch::nativert::detail {

using torch::nativert::Value;

class NestedSpec;

using NestedFlattenFn =
    void (*)(const c10::IValue&, const NestedSpec&, std::vector<c10::IValue>&);
using NestedUnflattenFn =
    c10::IValue (*)(std::vector<c10::IValue>, const nlohmann::json&);

using ContextLoadFn = nlohmann::json (*)(std::string_view);

using NestedMapFn = c10::function_ref<c10::IValue(const c10::IValue&)>;
using NestedMapNoReturnFn =
    c10::function_ref<void(const c10::IValue&, const Value*)>;

using IValueApplyFn =
    void (*)(NestedMapNoReturnFn, const c10::IValue&, const NestedSpec&);

nlohmann::json defaultContextLoadFn(std::string_view);

struct NodeDef {
  NestedFlattenFn flattenFn;
  NestedUnflattenFn unflattenFn;
  IValueApplyFn ivalueApplyFn;

  ContextLoadFn contextLoadFn = defaultContextLoadFn;
};

class NestedSpec {
 public:
  // Leaf node.
  NestedSpec(const Value* value = nullptr, bool isUsed = true)
      : numIValues_(1), value_(value), isUsed_(isUsed) {}

  // Non leaf node.
  NestedSpec(
      std::string_view uniformName,
      nlohmann::json context,
      std::vector<NestedSpec> children,
      NodeDef nodeDefCache);

  bool isIValue() const {
    return !uniformName_;
  }

  std::string_view uniformName() const {
    TORCH_CHECK(uniformName_);
    return uniformName_.value();
  }

  const nlohmann::json& context() const {
    return context_;
  }

  const std::vector<c10::IValue>& contextKeys() const {
    return contextKeys_;
  }

  const auto& children() const {
    return children_;
  }

  const NestedSpec& children(size_t i) const {
    return children_[i];
  }

  const NodeDef& nodeDefCache() const {
    return nodeDefCache_;
  }

  size_t numIValues() const {
    return numIValues_;
  }

  bool allIValues() const {
    return allIValues_;
  }

  c10::TypePtr toAtenType() const;

  bool isUsed() const {
    return isUsed_;
  }

  const Value* value() const {
    return value_;
  }

 private:
  // Only non leaf nodes have names.
  // Examples of uniform name: "builtins.tuple", "builtins.dict".
  std::optional<std::string> uniformName_;
  nlohmann::json context_;
  std::vector<NestedSpec> children_;

  std::vector<c10::IValue> contextKeys_;

  // Cached fields.
  NodeDef nodeDefCache_;
  size_t numIValues_;
  bool allIValues_ = true;

  const Value* value_;
  bool isUsed_;
};

void registerPytreeNode(std::string_view typeName, NodeDef nodeDef);

// Serialized json tree spec should be dumped from treespec_dumps() in
// torch.utils._pytree directly .
NestedSpec nestedSpecLoads(
    std::string_view json,
    const std::vector<const Value*>& values);

c10::IValue nestedUnflatten(
    std::vector<c10::IValue> ivalues,
    const NestedSpec& spec);

std::vector<c10::IValue> nestedFlatten(
    const c10::IValue& nested,
    const NestedSpec& spec);

std::vector<c10::IValue> nestedFlattenFromArgs(
    const std::vector<c10::IValue>& args,
    const std::unordered_map<std::string, c10::IValue>& kwargs,
    const NestedSpec& spec);

std::vector<at::Tensor> nestedFlattenToTensorList(
    const c10::IValue& nested,
    const NestedSpec& spec);

c10::IValue nestedMap(
    NestedMapFn f,
    const c10::IValue& nested,
    const NestedSpec& spec);

c10::IValue TORCH_API argsToIValue(
    const std::vector<c10::IValue>& args,
    const std::unordered_map<std::string, c10::IValue>& kwargs);

std::
    pair<std::vector<c10::IValue>, std::unordered_map<std::string, c10::IValue>>
    nestedMapArgs(
        NestedMapFn f,
        const std::vector<c10::IValue>& args,
        const std::unordered_map<std::string, c10::IValue>& kwargs,
        const NestedSpec& spec);

void ivalueApply(
    NestedMapNoReturnFn f,
    const c10::IValue& nested,
    const NestedSpec& spec);

void ivalueApplyFromArgs(
    NestedMapNoReturnFn fn,
    const std::vector<c10::IValue>& args,
    const std::unordered_map<std::string, c10::IValue>& kwargs,
    const NestedSpec& spec);

} // namespace torch::nativert::detail
