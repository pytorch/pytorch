#include <torch/csrc/lazy/core/trie.h>

#include <fstream>
#include <sstream>
#include <torch/csrc/lazy/core/hash.h>
#include <torch/csrc/lazy/core/internal_ops/ltc_ops.h>
#include <torch/csrc/lazy/core/ir_metadata.h>
#include "utils/memory.h"

namespace torch {
namespace lazy {
namespace {
void TraverseTrie(TrieNode* node, std::stringstream& ss) {
  if (!node) {
    return;
  }
  for (auto &successor : node->successors) {
    if (node->ir_node) {
      ss << node->unique_id << "[label=\"" << node->ir_node->op().ToString() << "\"]\n";
    }
    ss << node->unique_id << " -> " << successor->unique_id << "\n";
    TraverseTrie(successor.get(), ss);
  }
}
}

Trie* Trie::Get() {
  static Trie* trie = new Trie();
  return trie;
}

Trie::Trie() : root_(std::make_unique<TrieNode>()), current_(root_.get()) {
}

TrieNode* Trie::Current() const {
  return current_;
}

void Trie::SetCurrent(std::deque<std::unique_ptr<TrieNode>>::iterator iter) {
  auto& successors = current_->successors;
  current_ = (*iter).get();

  // Move *it to the front of the queue to achieve a LRU cache behavior
  if (iter != successors.begin()) {
    successors.push_front(std::move(*iter));
    successors.erase(iter);
  }
}

void Trie::ResetCurrent() {
  current_ = root_.get();
}

void Trie::Insert(NodePtr ir_node) {
  TORCH_CHECK(current_);
  if (!current_->successors.empty()) {
    TORCH_LAZY_COUNTER("ForkTrie", 1);
  }
  current_->successors.push_front(std::make_unique<TrieNode>(std::move(ir_node)));
  current_ = current_->successors.front().get();
}

void Trie::DumpToDotFile(const std::string& file_name) {
  std::stringstream ss;
  ss << "digraph G {\n";
  TraverseTrie(root_.get(), ss);
  ss << "}\n";

  std::ofstream graph_file(file_name);
  graph_file << ss.str();
}

} // namespace lazy
} // namespace torch
