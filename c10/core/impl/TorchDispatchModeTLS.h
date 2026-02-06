#pragma once

#include <c10/core/SafePyObject.h>
#include <c10/macros/Export.h>

namespace c10::impl {

enum class TorchDispatchModeKey : int8_t { FAKE, PROXY, FUNCTIONAL, NUM_MODE_KEYS };

using PyObject_TorchDispatchMode = SafePyObjectT<TorchDispatchModeKey>;

// Static dispatch state for optimized tracing
// When enabled, we pre-compute the dispatch chain and avoid TLS manipulation
// on every op, significantly reducing cold-start overhead.
struct C10_API StaticDispatchState {
  bool enabled = false;
  // Pre-computed dispatch chain (ordered from highest to lowest priority)
  std::vector<std::shared_ptr<PyObject_TorchDispatchMode>> dispatch_chain;
  // Current position in chain (decrements as we dispatch through modes)
  int64_t current_idx = -1;

  void reset() {
    enabled = false;
    dispatch_chain.clear();
    current_idx = -1;
  }
};

struct C10_API TorchDispatchModeTLS {
  // This API is NOT invariant safe.
  // It must not take in an infra mode that uses TorchDispatchModeKey
  // If you're pushing an infra mode onto the stack, we expect
  // you to use set_mode
  static void push_non_infra_mode_onto_stack(std::shared_ptr<PyObject_TorchDispatchMode> mode);
  // Pops the top mode of the stack,
  // giving precedence to user modes before attempting to pop
  // any infra modes
  static const std::shared_ptr<PyObject_TorchDispatchMode> pop_stack();
  // Returns the highest-priority infra mode on the stack,
  // along with its mode key.
  static const std::tuple<std::shared_ptr<PyObject_TorchDispatchMode>, TorchDispatchModeKey> pop_highest_infra_mode();

  static const std::shared_ptr<PyObject_TorchDispatchMode>& get_stack_at(int64_t idx);
  static int64_t stack_len();

  static const std::optional<std::shared_ptr<PyObject_TorchDispatchMode>> get_mode(TorchDispatchModeKey mode_key);
  static const std::optional<std::shared_ptr<PyObject_TorchDispatchMode>> unset_mode(TorchDispatchModeKey mode_key);
  static void set_mode(const std::shared_ptr<PyObject_TorchDispatchMode>& mode, TorchDispatchModeKey mode_key);

  static const TorchDispatchModeTLS& get_state();
  static void set_state(TorchDispatchModeTLS state);

  static bool any_modes_set(bool skip_infra_modes = false);

  // Static dispatch optimization APIs
  // Enable static dispatch by snapshotting the current mode stack
  static void enable_static_dispatch();
  // Disable static dispatch and restore normal TLS-based dispatch
  static void disable_static_dispatch();
  // Check if static dispatch is currently enabled
  static bool is_static_dispatch_enabled();
  // Get the current mode in static dispatch without TLS manipulation
  // Returns nullptr if no more modes in the chain
  static std::shared_ptr<PyObject_TorchDispatchMode> get_current_static_mode();
  // Advance to the next mode in static dispatch chain
  static void advance_static_dispatch();
  // Reset static dispatch index (for re-dispatch from top)
  static void reset_static_dispatch_index();

 private:
  std::vector<std::shared_ptr<PyObject_TorchDispatchMode>> stack_;
  // Users are allowed to push multiple ProxyTorchDispatchMode objects onto the
  // stack
  // However, we only allow a single FakeTensorMode onto the stack at a time
  // (Pushing additional FakeTensorModes onto the stack is a no-op)
  std::array<std::optional<std::shared_ptr<PyObject_TorchDispatchMode>>,
             static_cast<size_t>(TorchDispatchModeKey::NUM_MODE_KEYS)>
      infra_modes_;
};

C10_API bool dispatch_mode_enabled();

C10_API std::string to_string(TorchDispatchModeKey mode_key);

} // namespace c10::impl
