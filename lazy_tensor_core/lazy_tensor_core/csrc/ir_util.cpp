#include "lazy_tensor_core/csrc/ir_util.h"

#include "lazy_tensors/computation_client/debug_macros.h"

namespace torch_lazy_tensors {
namespace ir {

std::vector<const torch::lazy::Node*> Util::ComputePostOrder(const torch::lazy::Node* node,
                                                EmissionMap* emap) {
  std::vector<const torch::lazy::Node*> post_order;
  std::vector<const torch::lazy::Node*> queue;
  queue.push_back(node);
  while (!queue.empty()) {
    node = queue.back();
    auto it = emap->find(node);
    if (it == emap->end()) {
      (*emap)[node] = kEmitting;

      for (auto& output : node->operands()) {
        auto oit = emap->find(output.node);
        if (oit == emap->end()) {
          queue.push_back(output.node);
        } else if (oit->second == kEmitting) {
          LTC_ERROR() << "Graph loop found at " << *output.node;
        }
      }
    } else if (it->second == kEmitting) {
      for (auto& output : node->operands()) {
        auto oit = emap->find(output.node);
        LTC_CHECK(oit != emap->end() && oit->second == kEmitted)
            << "Graph loop found at " << *output.node;
      }
      (*emap)[node] = kEmitted;
      post_order.push_back(node);
      queue.pop_back();
    } else {
      LTC_CHECK_EQ(it->second, kEmitted);
      queue.pop_back();
    }
  }
  return post_order;
}

std::vector<const torch::lazy::Node*> Util::ComputePostOrder(
    lazy_tensors::Span<const torch::lazy::Node* const> nodes, EmissionMap* emap) {
  std::vector<const torch::lazy::Node*> post_order;
  for (auto node : nodes) {
    auto node_post_order = ComputePostOrder(node, emap);
    post_order.insert(post_order.end(), node_post_order.begin(),
                      node_post_order.end());
  }
  return post_order;
}

std::vector<const torch::lazy::Node*> Util::ComputePostOrder(
    lazy_tensors::Span<const torch::lazy::Node* const> nodes) {
  EmissionMap emap;
  return ComputePostOrder(nodes, &emap);
}

size_t Util::GetGraphSize(lazy_tensors::Span<const torch::lazy::Node* const> nodes) {
  std::vector<const torch::lazy::Node*> post_order = ComputePostOrder(nodes);
  return post_order.size();
}

}  // namespace ir
}  // namespace torch_lazy_tensors
