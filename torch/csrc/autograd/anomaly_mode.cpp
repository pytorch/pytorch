#include <c10/util/Backtrace.h>
#include <c10/util/Exception.h>
#include <torch/csrc/autograd/anomaly_mode.h>
#include <torch/csrc/autograd/function.h>
#include <mutex>

namespace torch {
namespace autograd {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
bool AnomalyMode::_enabled = false;

namespace {
std::mutex& get_anomaly_guard_lock() {
  static std::mutex anomaly_guard_lock{};
  return anomaly_guard_lock;
}

uint32_t& get_anomaly_counter() {
  static uint32_t counter = 0;
  return counter;
}
} // namespace

DetectAnomalyGuard::DetectAnomalyGuard() {
  TORCH_WARN_ONCE(
      "This mode should be enabled only for debugging as the different tests will slow down your program execution.");
  std::lock_guard<std::mutex> lock(get_anomaly_guard_lock());
  uint32_t& counter = get_anomaly_counter();
  counter++;
  AnomalyMode::set_enabled(true);
}

DetectAnomalyGuard::~DetectAnomalyGuard() {
  std::lock_guard<std::mutex> lock(get_anomaly_guard_lock());
  uint32_t& counter = get_anomaly_counter();
  counter--;
  AnomalyMode::set_enabled(counter > 0);
}

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
