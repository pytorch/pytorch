#pragma once

#include <c10/macros/Export.h>

namespace c10 {

// Structure used to pack all the thread local boolean
// flags used by autograd
struct C10_API AutogradState {
  static AutogradState& get_tls_state();
  static void set_tls_state(AutogradState state);

  AutogradState(
      bool grad_mode,
      bool inference_mode,
      bool fw_grad_mode,
      bool multithreading_enabled)
      : grad_mode_(grad_mode),
        inference_mode_(inference_mode),
        fw_grad_mode_(fw_grad_mode),
        multithreading_enabled_(multithreading_enabled),
        view_replay_enabled_(false) {}

  void set_grad_mode(bool enabled) {
    grad_mode_ = enabled;
  }

  void set_fw_grad_mode(bool enabled) {
    fw_grad_mode_ = enabled;
  }

  void set_inference_mode(bool enabled) {
    inference_mode_ = enabled;
  }

  void set_multithreading_enabled(bool multithreading_enabled) {
    multithreading_enabled_ = multithreading_enabled;
  }

  void set_view_replay_enabled(bool view_replay_enabled) {
    view_replay_enabled_ = view_replay_enabled;
  }

  bool get_grad_mode() const {
    return grad_mode_;
  }

  bool get_fw_grad_mode() const {
    return fw_grad_mode_;
  }

  bool get_inference_mode() const {
    return inference_mode_;
  }

  bool get_multithreading_enabled() const {
    return multithreading_enabled_;
  }

  bool get_view_replay_enabled() const {
    return view_replay_enabled_;
  }

 private:
  bool grad_mode_ : 1;
  bool inference_mode_ : 1;
  bool fw_grad_mode_ : 1;
  bool multithreading_enabled_ : 1;
  // NOLINTNEXTLINE(cppcoreguidelines-use-default-member-init)
  bool view_replay_enabled_ : 1;
};

} // namespace c10
