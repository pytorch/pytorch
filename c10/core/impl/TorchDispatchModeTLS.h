#pragma once

#include <c10/core/SafePyObject.h>
#include <c10/macros/Export.h>

namespace c10 {
namespace impl {

enum class TorchDispatchModeKey : int8_t {
  FUNCTIONAL = 0,
  PROXY = 1,
  FAKE = 2,
  NUM_MODE_KEYS = 3
};

struct C10_API TorchDispatchModeTLS {
  // This API is NOT invariant safe.
  // We expect that if your mode is one of the special modes that gets a mode
  // key associated with it, you will have supplied the correct mode key. This
  // invariant checking is done properly in torch/csrc/autograd/init.cpp (where
  // these API's are exposed to python).
  static void push_onto_stack(
      std::shared_ptr<SafePyObject> mode,
      c10::optional<TorchDispatchModeKey> mode_key = c10::nullopt);
  // Returns the highest-priority mode on the stack.
  // If the highest-priority mode has a ModeKey associated with it, we also
  // return the ModeKey.
  // If a mode key is passed in, we will pop the mode that corresponds to the
  // mode key. Otherwise, we will pop the top of the stack.
  static const std::tuple<
      std::shared_ptr<SafePyObject>,
      c10::optional<TorchDispatchModeKey>>
  pop_stack(c10::optional<TorchDispatchModeKey> maybe_mode_key = c10::nullopt);

  static const std::shared_ptr<SafePyObject>& get_stack_at(int64_t idx);
  static int64_t stack_len();

  static const c10::optional<std::shared_ptr<SafePyObject>> get_mode(
      TorchDispatchModeKey mode_key);
  static const c10::optional<std::shared_ptr<SafePyObject>> unset_mode(
      TorchDispatchModeKey mode_key);
  static void set_mode(
      const std::shared_ptr<SafePyObject>& mode,
      TorchDispatchModeKey mode_key);

  static const TorchDispatchModeTLS& get_state();
  static void set_state(TorchDispatchModeTLS state);

  static bool any_modes_set(bool skip_infra_modes = false);

 private:
  std::vector<std::shared_ptr<c10::SafePyObject>> stack_;
  // Users are allowed to push multiple ProxyTorchDispatchMode objects onto the
  // stack
  // However, we only allow asingle FakeTensorMode onto the stack at a time
  // (Pushing additional FakeTensorModes onto the stack is a no-op)
  std::array<
      c10::optional<std::shared_ptr<c10::SafePyObject>>,
      static_cast<size_t>(TorchDispatchModeKey::NUM_MODE_KEYS)>
      infra_modes_;
};

C10_API bool dispatch_mode_enabled(bool skip_infra_modes = false);

C10_API std::string to_string(TorchDispatchModeKey mode_key);

} // namespace impl
} // namespace c10
