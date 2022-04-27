#pragma once

#include <torch/csrc/lazy/core/ir.h>

namespace torch {
namespace lazy {

template <typename T>
const T* NodeCast(const Node* node, OpKind op) {
  if (op != node->op()) {
    return nullptr;
  }
#ifdef NDEBUG
  return static_cast<const T*>(node);
#else
  return &dynamic_cast<const T&>(*node);
#endif
}

// TODO(alanwaketan): Support r-value reference argument type.
template <typename T, typename... Args>
NodePtr MakeNode(Args&&... args) {
  return std::make_shared<T>(std::forward<Args>(args)...);
}

template <typename T, typename... Args>
NodePtr ReuseOrMakeNode(OpKind op, Args&&... args) {
  // TODO: add logic for ReuseNode later
  // NodePtr node = ReuseNode<T>(op, std::forward<Args>(args)...);
  NodePtr node = nullptr;
  if (!node) {
    node = MakeNode<T>(std::forward<Args>(args)...);
  }
  return node;
}

} // namespace lazy
} // namespace torch

