#include <torch/csrc/lazy/core/ir_util.h>

#include <c10/util/Logging.h>

namespace torch {
namespace lazy {

std::vector<Node*> Util::ComputePostOrder(const Node* node, EmissionMap* emap) {
  std::vector<Node*> post_order;
  std::vector<Node*> queue;
  // std::vector<const T> to c10::ArrayRef<T> conversion is not supported,
  // so we need to drop const in the return vector and use const_cast here.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  queue.push_back(const_cast<Node*>(node));
  while (!queue.empty()) {
    node = queue.back();
    auto it = emap->find(node);
    if (it == emap->end()) {
      (*emap)[node] = kEmitting;
      for (auto& output : node->operands()) {
        auto oit = emap->find(output.node);
        if (oit == emap->end()) {
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          queue.push_back(const_cast<Node*>(output.node));
        } else {
          TORCH_CHECK(
              oit->second != kEmitting,
              "Graph loop found at ",
              output.node->ToString());
        }
      }
    } else if (it->second == kEmitting) {
      for (auto& output : node->operands()) {
        auto oit = emap->find(output.node);
        TORCH_CHECK(
            oit != emap->end() && oit->second == kEmitted,
            "Graph loop found at ",
            output.node->ToString());
      }
      (*emap)[node] = kEmitted;
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      post_order.push_back(const_cast<Node*>(node));
      queue.pop_back();
    } else {
      TORCH_CHECK(it->second == kEmitted);
      queue.pop_back();
    }
  }
  return post_order;
}

std::vector<Node*> Util::ComputePostOrder(
    c10::ArrayRef<Node*> nodes,
    EmissionMap* emap) {
  std::vector<Node*> post_order;
  for (auto node : nodes) {
    auto node_post_order = ComputePostOrder(node, emap);
    post_order.insert(
        post_order.end(), node_post_order.begin(), node_post_order.end());
  }
  return post_order;
}

std::vector<Node*> Util::ComputePostOrder(c10::ArrayRef<Node*> nodes) {
  EmissionMap emap;
  return ComputePostOrder(nodes, &emap);
}

size_t Util::GetGraphSize(c10::ArrayRef<Node*> nodes) {
  return ComputePostOrder(nodes).size();
}

} // namespace lazy
} // namespace torch
