#pragma once

#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>
#include <torch/csrc/lazy/backend/backend_interface.h>
#include <torch/csrc/lazy/core/config.h>
#include <torch/csrc/lazy/core/ir.h>
#include <torch/csrc/lazy/core/tensor.h>
#include <torch/csrc/lazy/core/trie.h>
#include <vector>

// This file is part of the backend interface. So, ops shouldn't be added or
// removed without due process The exception to this being the view ops which
// will be removed soon pending functionalization

namespace torch {
namespace lazy {

template <typename T, typename... Args>
NodePtr ReuseNode(Args&&... args) {
  if (FLAGS_torch_lazy_reuse_ir) {
    return LookupNodeFromTrieCache<T>(std::forward<Args>(args)...);
  }
  return nullptr;
}

// Caching an IR node into TrieCache
static inline void CacheNode(NodePtr node) {
  if (FLAGS_torch_lazy_reuse_ir) {
    TrieCache::Get()->Insert(std::move(node));
  }
}

template <typename T, typename... Args>
NodePtr MakeNode(Args&&... args) {
  return std::make_shared<T>(std::forward<Args>(args)...);
}

// op is passed in for a more efficient node casting, see the implementation of
// NodeCast
template <typename T, typename... Args>
NodePtr ReuseOrMakeNode(Args&&... args) {
  NodePtr node = ReuseNode<T>(std::forward<Args>(args)...);
  if (!node) {
    node = MakeNode<T>(std::forward<Args>(args)...);
    CacheNode(node);
  }
  return node;
}

struct IrBuilder {
  virtual NodePtr MakeDeviceData(
      const std::shared_ptr<BackendData>& data) const = 0;
  virtual NodePtr MakeScalar(
      const at::Scalar& value,
      const at::ScalarType& type) const = 0;
  virtual NodePtr MakeExpand(
      const Value& input0,
      const std::vector<int64_t>& size,
      const bool& is_scalar_expand) const = 0;
  virtual NodePtr MakeCast(
      const Value& input0,
      const at::ScalarType& dtype,
      const c10::optional<at::ScalarType>& stype = c10::nullopt) const = 0;
  virtual NodePtr MakeTensorList(const OpList& inputs) const = 0;
  virtual NodePtr MakeGeneric(
      const OpKind& op,
      const OpList& operands,
      const Shape& shape,
      const size_t& num_outputs = 1,
      const hash_t& hash_seed = static_cast<uint32_t>(0x5a2d296e9)) const = 0;

  // dynamic ir nodes
  virtual NodePtr MakeSizeNode(const Value& input, size_t dim) const = 0;
  virtual NodePtr MakeSizeAdd(const Value& a, const Value& b) const = 0;
  virtual NodePtr MakeSizeMul(const Value& a, const Value& b) const = 0;
  virtual NodePtr MakeSizeDiv(const Value& a, const Value& b) const = 0;

  virtual ~IrBuilder() = default;
};

static inline NodePtr MakeDeviceData(const std::shared_ptr<BackendData>& data) {
  return getIrBuilder()->MakeDeviceData(data);
}
static inline NodePtr MakeScalar(
    const at::Scalar& value,
    const at::ScalarType& type) {
  return getIrBuilder()->MakeScalar(value, type);
}
static inline NodePtr MakeExpand(
    const Value& input0,
    const std::vector<int64_t>& size,
    const bool& is_scalar_expand) {
  return getIrBuilder()->MakeExpand(input0, size, is_scalar_expand);
}
static inline NodePtr MakeCast(
    const Value& input0,
    const at::ScalarType& dtype,
    const c10::optional<at::ScalarType>& stype = c10::nullopt) {
  return getIrBuilder()->MakeCast(input0, dtype, stype);
}
static inline NodePtr MakeTensorList(const OpList& inputs) {
  return getIrBuilder()->MakeTensorList(inputs);
}
static inline NodePtr MakeGeneric(
    const OpKind& op,
    const OpList& operands,
    const Shape& shape,
    const size_t& num_outputs = 1,
    const hash_t& hash_seed = static_cast<uint32_t>(0x5a2d296e9)) {
  return getIrBuilder()->MakeGeneric(
      op, operands, shape, num_outputs, hash_seed);
}

// dynamic ir nodes
static inline NodePtr MakeSizeNode(const Value& input, size_t dim) {
  return getIrBuilder()->MakeSizeNode(input, dim);
}
static inline NodePtr MakeSizeAdd(const Value& a, const Value& b) {
  return getIrBuilder()->MakeSizeAdd(a, b);
}
static inline NodePtr MakeSizeMul(const Value& a, const Value& b) {
  return getIrBuilder()->MakeSizeAdd(a, b);
}
static inline NodePtr MakeSizeDiv(const Value& a, const Value& b) {
  return getIrBuilder()->MakeSizeDiv(a, b);
}

inline Value GetSymIntValue(c10::SymInt a) {
  return Value(
      a.is_symbolic()
          ? dynamic_cast<torch::lazy::SymNodeImpl*>(a.toSymNodeImpl().get())
                ->node_
          : MakeScalar(a.as_int_unchecked(), at::kLong),
      0);
}

// TODO: this should return Value
inline std::vector<int64_t> GetSymIntArrayRefValue(c10::SymIntArrayRef arr) {
  std::vector<int64_t> r;
  for (const auto& a : arr) {
    r.emplace_back(a.expect_int());
  }
  return r;
}

} // namespace lazy
} // namespace torch
