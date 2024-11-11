#include <torch/csrc/lazy/core/trie.h>

#include <torch/csrc/lazy/core/hash.h>
#include <torch/csrc/lazy/core/internal_ops/ltc_ops.h>
#include <torch/csrc/lazy/core/ir_metadata.h>
#include <torch/csrc/lazy/core/metrics.h>
#include <fstream>
#include <sstream>

namespace torch::lazy {
namespace {

void TraverseTrie(TrieNode* node, std::stringstream& ss) {
  if (!node) {
    return;
  }
  if (node->ir_node) {
    ss << node->unique_id << "[label=\"" << node->ir_node->op().ToString()
       << ", " << node->hit_counter << " hits\"]\n";
  }
  for (auto& successor : node->successors) {
    ss << node->unique_id << " -> " << successor->unique_id << "\n";
    TraverseTrie(successor.get(), ss);
  }
}
} // namespace

TrieCache* TrieCache::Get() {
  static thread_local TrieCache* trie = new TrieCache();
  return trie;
}

TrieCache::TrieCache()
    : root_(std::make_shared<TrieNode>()), current_(root_.get()) {}

TrieNode* TrieCache::Current() const {
  return current_;
}

void TrieCache::SetCurrent(
    std::list<std::shared_ptr<TrieNode>>::iterator& iter) {
  auto& successors = current_->successors;
  // Update current_ before iter gets destroyed
  current_ = (*iter).get();

  // Insert this node to the front of its parent's successor list
  if (iter != successors.begin()) {
    successors.push_front(std::move(*iter));
    successors.erase(iter);
  }
}

void TrieCache::ResetCurrent() {
  current_ = root_.get();
}

void TrieCache::Insert(NodePtr ir_node) {
  TORCH_CHECK(current_);
  if (!current_->successors.empty()) {
    TORCH_LAZY_COUNTER("TrieForked", 1);
  }
  auto new_node = std::make_shared<TrieNode>(std::move(ir_node));
  current_->successors.push_front(std::move(new_node));
  // Update current_ to the newly inserted node
  current_ = current_->successors.front().get();
}

void TrieCache::Clear() {
  ResetCurrent();
  // Clear at the root level should be sufficient because all the nodes
  // are created as shared_ptr.
  root_->successors.clear();
}

void TrieCache::DumpToDotFile(const std::string& file_name) {
  std::stringstream ss;
  ss << "digraph G {\n";
  TraverseTrie(root_.get(), ss);
  ss << "}\n";

  std::ofstream graph_file(file_name);
  graph_file << ss.str();
}

} // namespace torch::lazy
