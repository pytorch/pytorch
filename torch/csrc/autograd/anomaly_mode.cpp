#include <c10/util/Backtrace.h>
#include <c10/util/Exception.h>
#include <torch/csrc/autograd/anomaly_mode.h>
#include <torch/csrc/autograd/function.h>

namespace torch {
namespace autograd {

bool AnomalyMode::_enabled = false;

AnomalyMetadata::~AnomalyMetadata() = default;

void AnomalyMetadata::store_stack() {
  traceback_ = c10::get_backtrace(/* frames_to_skip */ 1);
}

void AnomalyMetadata::print_stack(const std::string& current_node_name) {
  TORCH_WARN(
      "Error detected in ",
      current_node_name,
      ". ",
      "Traceback of forward call that caused the error:\n",
      traceback_);

  auto& cur_parent = parent_;
  // if there is no "parent_" in metadata, then it means this metadata's node
  // is the root and stop printing the traceback
  while (cur_parent) {
    auto parent_metadata = cur_parent->metadata();
    TORCH_WARN(
        "\n\n",
        "Previous calculation was induced by ",
        cur_parent->name(),
        ". "
        "Traceback of forward call that induced the previous calculation:\n",
        parent_metadata->traceback_);
    // get the parent of this node, if this node is a root, pyparent is simply
    // null
    cur_parent = parent_metadata->parent_;
  }
}

void AnomalyMetadata::assign_parent(const std::shared_ptr<Node>& parent_node) {
  parent_ = parent_node;
}

} // namespace autograd
} // namespace torch
