#include <torch/csrc/lazy/core/ir_util.h>

#include <stack>

#include <c10/util/Logging.h>

namespace torch::lazy {

std::vector<const Node*> Util::ComputePostOrder(
    const Node* node,
    EmissionMap* emap) {
  std::vector<const Node*> post_order;
  std::stack<const Node*> node_stack;
  node_stack.push(node);
  while (!node_stack.empty()) {
    node = node_stack.top();
    auto it = emap->find(node);
    if (it == emap->end()) {
      (*emap)[node] = kEmitting;
      for (auto& output : node->operands()) {
        auto oit = emap->find(output.node);
        if (oit == emap->end()) {
          node_stack.push(output.node);
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
      post_order.push_back(node);
      node_stack.pop();
    } else {
      TORCH_CHECK(it->second == kEmitted);
      node_stack.pop();
    }
  }
  return post_order;
}

std::vector<const Node*> Util::ComputePostOrder(
    c10::ArrayRef<const Node*> nodes,
    EmissionMap* emap) {
  std::vector<const Node*> post_order;
  for (auto node : nodes) {
    auto node_post_order = ComputePostOrder(node, emap);
    post_order.insert(
        post_order.end(), node_post_order.begin(), node_post_order.end());
  }
  return post_order;
}

std::vector<const Node*> Util::ComputePostOrder(
    c10::ArrayRef<const Node*> nodes) {
  EmissionMap emap;
  return ComputePostOrder(nodes, &emap);
}

size_t Util::GetGraphSize(c10::ArrayRef<const Node*> nodes) {
  return ComputePostOrder(nodes).size();
}

} // namespace torch::lazy
