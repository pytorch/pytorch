#pragma once

#include <atomic>
#include <deque>

#include <c10/core/ScalarType.h>
#include <torch/csrc/lazy/core/ir.h>
#include <torch/csrc/lazy/core/metrics.h>

namespace torch {
namespace lazy {

struct TORCH_API TrieNode {
  static size_t GetNextUniqueId() {
    static thread_local size_t id_generator = 0;
    return id_generator++;
  }

  size_t unique_id;
  NodePtr ir_node;
  std::deque<std::shared_ptr<TrieNode>> successors;

  TrieNode() : unique_id(GetNextUniqueId()), ir_node(nullptr) {}
  explicit TrieNode(NodePtr node)
      : unique_id(GetNextUniqueId()), ir_node(std::move(node)) {}
};

class TORCH_API TrieCache {
 public:
  static TrieCache* Get();

  TrieNode* Current() const;
  // Take an iterator as the input because we want to move the corresponding
  // node in the successor list to achieve a LRU caching effect
  void SetCurrent(std::deque<std::shared_ptr<TrieNode>>::iterator iter);
  // Used in MarkStep to indicate the end of one tracing
  void ResetCurrent();

  // Create a new TrieNode for ir_node and insert into the TrieCache
  void Insert(NodePtr ir_node);

  // Clear all TrieCache nodes
  void Clear();

  void DumpToDotFile(const std::string& file_name);

 private:
  TrieCache();

  std::shared_ptr<TrieNode> root_;
  TrieNode* current_;
};

template <typename T, typename... Args>
NodePtr LookupNodeFromTrieCache(Args&&... args) {
  auto& successors = TrieCache::Get()->Current()->successors;
  for (auto it = successors.begin(); it != successors.end(); it++) {
    NodePtr ir_node = (*it)->ir_node;
    const T* concrete_node = NodeCast<T>(ir_node.get());
    if (concrete_node && concrete_node->Equal(std::forward<Args>(args)...)) {
      TORCH_LAZY_COUNTER("IrNodeReused::" + std::string(typeid(T).name()), 1);
      TrieCache::Get()->SetCurrent(it);
      return ir_node;
    }
  }
  return nullptr;
}

} // namespace lazy
} // namespace torch
