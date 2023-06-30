#pragma once

#include <c10/core/SafePyObject.h>
#include <c10/macros/Export.h>

namespace c10 {
namespace impl {

struct C10_API TorchDispatchModeTLS {
  static void push_onto_stack(std::shared_ptr<SafePyObject> mode);
  static const std::shared_ptr<SafePyObject> pop_stack();
  static const std::shared_ptr<SafePyObject>& get_stack_at(int64_t idx);
  static int64_t stack_len();

  // Gets the highest-priority mode, or none if there are no active modes
  static const c10::optional<std::shared_ptr<SafePyObject>> maybe_highest_mode();

  static const c10::optional<std::shared_ptr<SafePyObject>> get_fake_mode();
  static const std::shared_ptr<SafePyObject> unset_fake_mode();
  static void set_fake_mode(std::shared_ptr<SafePyObject> mode);

  static const c10::optional<std::shared_ptr<SafePyObject>> get_proxy_mode();
  static const std::shared_ptr<SafePyObject> unset_proxy_mode();
  static void set_proxy_mode(std::shared_ptr<SafePyObject> mode);

  static const TorchDispatchModeTLS& get_state();
  static void set_state(TorchDispatchModeTLS state);

 private:
  std::vector<std::shared_ptr<c10::SafePyObject>> stack_;
  // Users are allowed to push multiple ProxyTorchDispatchMode objects onto the
  // stack
  c10::optional<std::shared_ptr<c10::SafePyObject>> proxy_mode_;
  // However, we only allow asingle FakeTensorMode onto the stack at a time
  // (Pushing additional FakeTensorModes onto the stack is a no-op)
  c10::optional<std::shared_ptr<c10::SafePyObject>> fake_mode_;
};

C10_API bool dispatch_mode_enabled(bool skip_proxy_and_fake = false);

} // namespace impl
} // namespace c10
