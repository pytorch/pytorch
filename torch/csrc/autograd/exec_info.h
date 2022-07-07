#pragma once

// Exec info has a bit complicated semantics. If it's empty, it means the task
// is run in a "default" mode, which means that all next_edges we encounter
// should get executed. If it's not empty, only functions that have an entry
// and this entry has needed == True should be executed. exec_info is only empty
// when the graph is executed via .backward() and the inputs parameter is not passed.
// Otherwise, when executed through .grad(), or when inputs arg is specified for
// .backward(), exec_info will be non-empty.
//

#include <ATen/Tensor.h>
#include <vector>

namespace torch { namespace autograd {

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct ExecInfo {
  struct Capture {
    Capture(const Capture&) = delete;
    Capture(Capture&&) = default;

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    Capture(int input_idx, int output_idx)
        : input_idx_(input_idx), output_idx_(output_idx) {}
    int input_idx_; // within Node inputs
    int output_idx_; // within the output vector of a GraphTask

    // This hook will be executed after a grad is captured. The captured
    // grad will be replaced by the return value of the hook.
    struct GradCaptureHook {
      virtual ~GradCaptureHook() = default;
      virtual at::Tensor operator()(const at::Tensor& grad) = 0;
    };
    // The hooks will be called one by one in the order as they were added.
    // The input grad of a hook will be the output of its preceding hook. The
    // first hook will take the captured grad as the input. The output of the
    // last hook will replace the captured grad.
    std::vector<std::unique_ptr<GradCaptureHook>> hooks_;
  };

  bool should_execute() const {
    return needed_ || captures_;
  }

  bool needed_ = false;
  std::unique_ptr<std::vector<Capture>> captures_;
};

std::unordered_map<Node*, ExecInfo>* get_current_graph_task_exec_info();

}} // namespace torch::autograd
