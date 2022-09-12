#pragma once

#include <atomic>
#include <list>

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
  size_t hit_counter;
  NodePtr ir_node;
  std::list<std::shared_ptr<TrieNode>> successors;

  TrieNode() : unique_id(GetNextUniqueId()), hit_counter(0), ir_node(nullptr) {}
  explicit TrieNode(NodePtr node)
      : unique_id(GetNextUniqueId()),
        hit_counter(0),
        ir_node(std::move(node)) {}
};

class TORCH_API TrieCache {
 public:
  static TrieCache* Get();

  TrieNode* Current() const;
  // Take an iterator as the input because we want to move the corresponding
  // node in the successor list to achieve a LRU caching effect
  void SetCurrent(std::list<std::shared_ptr<TrieNode>>::iterator& iter);
  // Used in MarkStep to indicate the end of one tracing
  void ResetCurrent();

  // Create a new TrieNode for ir_node and insert into the TrieCache
  void Insert(NodePtr ir_node);

  // Clear all TrieCache nodes
  // TODO: Because we don't expect user to explicitly call this function via
  // a Python API, we may need to introduce a threshold on the size of the cache
  // to avoid holding tensors for too long.
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
    if (concrete_node &&
        concrete_node->CanBeReused(std::forward<Args>(args)...)) {
      TORCH_LAZY_COUNTER(
          "IrNodeReused_" + c10::demangle((typeid(T).name())), 1);
      (*it)->hit_counter++;
      TrieCache::Get()->SetCurrent(it);
      return ir_node;
    }
  }
  return nullptr;
}

} // namespace lazy
} // namespace torch
