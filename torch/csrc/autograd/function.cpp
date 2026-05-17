#include <torch/csrc/autograd/function.h>

#include <c10/util/ThreadLocal.h>
#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/autograd/variable.h>

#include <ATen/ATen.h>

#include <string>
#include <utility>
#include <vector>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/zeros.h>
#endif

namespace torch::autograd {

// The current evaluating node. This is useful to assign the current node as a
// parent of new nodes created during the evaluation of this node in anomaly
// mode.
C10_DEFINE_TLS_static(c10::intrusive_ptr<Node>, tls_current_evaluating_node);
#define current_evaluating_node (tls_current_evaluating_node.get())

NodeGuard::NodeGuard(c10::intrusive_ptr<Node> node)
    : last_evaluating_node_(std::move(current_evaluating_node)) {
  current_evaluating_node = std::move(node);
}
NodeGuard::~NodeGuard() {
  // restore the previous evaluating node
  current_evaluating_node = std::move(last_evaluating_node_);
}

c10::intrusive_ptr<Node> get_current_node() {
  return current_evaluating_node;
}

void Node::assign_parent() {
  metadata()->assign_parent(current_evaluating_node);
}

auto Node::name() const -> std::string {
  return c10::demangle(typeid(*this).name());
}

auto Node::forward_op_name() const -> std::string {
  auto n = name();
  // Strip "Backward<N>" suffix to get the forward op name.
  auto pos = n.rfind("Backward");
  if (pos == std::string::npos) {
    return n;
  }
  // Verify everything after "Backward" is digits (e.g., "Backward0").
  auto suffix_start = pos + 8;
  for (size_t i = suffix_start; i < n.size(); ++i) {
    if (!std::isdigit(static_cast<unsigned char>(n[i]))) {
      return n;
    }
  }
  // Keep the numeric suffix if it is not "0" (e.g., "AddBackward1" → "Add1").
  auto suffix = n.substr(suffix_start);
  if (suffix == "0" || suffix.empty()) {
    return n.substr(0, pos);
  }
  return n.substr(0, pos) + suffix;
}

bool Node::task_should_compute_output(size_t output_edge_index) const {
  TORCH_CHECK(output_edge_index < num_outputs(), "Index out of range");
  const auto& next = next_edges_[output_edge_index];
  if (next.is_valid()) {
    const auto exec_info = get_current_graph_task_exec_info();
    if (exec_info && !exec_info->empty()) {
      auto it = exec_info->find(next.function.get());
      if (it == exec_info->end() || !it->second.should_execute()) {
        return false;
      }
    }
    return true;
  }
  return false;
}

AnomalyMetadata* Node::metadata() noexcept {
  if (!anomaly_metadata_) {
    anomaly_metadata_ = Engine::get_default_engine().make_anomaly_metadata();
  }
  return anomaly_metadata_.get();
}

// Iteratively release child nodes to prevent stack overflow on deletion
// of deep computation graphs. See
// https://github.com/pytorch/pytorch/issues/5534
static void gatherFunctions(
    Node* func,
    std::vector<c10::intrusive_ptr<Node>>& stack) {
  func->release_variables();

  for (auto& edge : func->next_edges()) {
    if (edge.function.use_count() == 1) {
      stack.emplace_back(std::move(edge.function));
    } else {
      edge.function.reset();
    }
  }
}

static void releaseGraphIteratively(Node* node) {
  std::vector<c10::intrusive_ptr<Node>> stack;
  gatherFunctions(node, stack);
  while (!stack.empty()) {
    auto func = std::move(stack.back());
    stack.pop_back();
    gatherFunctions(func.get(), stack);
  }
}

void Node::release_resources() {
  releaseGraphIteratively(this);
  pre_hooks_.clear();
  post_hooks_.clear();
  tensor_pre_hooks_.clear();
  retains_grad_hooks_.clear();
  anomaly_metadata_.reset();
}

Node::~Node() {
  releaseGraphIteratively(this);
}

at::Tensor TypeAndSize::zeros() {
  return at::zeros_symint(sym_sizes, options);
}

} // namespace torch::autograd
