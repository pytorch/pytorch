#pragma once

#include <deque>
#include <atomic>

#include <c10/core/ScalarType.h>

namespace torch {
namespace lazy {

class Node;
using NodePtr = std::shared_ptr<Node>;

struct TORCH_API TrieNode {
  static int64_t GetNextUniqueId() {
    static size_t thread_local id_generator = 0;
    return id_generator++;
  }

  int64_t unique_id;
  NodePtr ir_node;
  std::deque<std::unique_ptr<TrieNode>> successors;

  TrieNode() : unique_id(GetNextUniqueId()), ir_node(nullptr) { }
  explicit TrieNode(NodePtr node) : unique_id(GetNextUniqueId()), ir_node(std::move(node)) { }
};

class TORCH_API Trie {
public:
  static Trie* Get();

  TrieNode* Current() const;
  void SetCurrent(std::deque<std::unique_ptr<TrieNode>>::iterator iter);
  void ResetCurrent();

  // Create a new TrieNode for ir_node and insert into the Trie
  void Insert(NodePtr ir_node);

  void DumpToDotFile(const std::string& file_name);

private:
  Trie();

  std::unique_ptr<TrieNode> root_;
  TrieNode* current_;
};

} // namespace lazy
} // namespace torch
