#pragma once

#include <c10/macros/Export.h>
#include <c10/util/Optional.h>
#include <c10/util/python_stub.h>
#include <stack>
#include <string>

#include <utility>

namespace at {

namespace impl {

struct TORCH_API SavedTensorDefaultHooksTLS {
  // PyObject is defined in c10/util/python_stub.h
  std::stack<std::pair<PyObject*, PyObject*>> stack;

  // See NOTE: [Disabling SavedTensorDefaultHooks] for context
  // NOTE: [disabled_error_message invariant]
  // disabled_error_message is nullopt IFF Saved Tensor hooks is enabled
  // We did this for efficiency (so we didn't have to keep a separate bool
  // around)
  c10::optional<std::string> disabled_error_message;
};

} // namespace impl

struct TORCH_API SavedTensorDefaultHooks {
  static void push_hooks(PyObject* pack_hook, PyObject* unpack_hook);
  static void pop_hooks();
  static std::pair<PyObject*, PyObject*> get_hooks();
  static void lazy_initialize();
  static std::stack<std::pair<PyObject*, PyObject*>> get_stack();
  static void set_stack(std::stack<std::pair<PyObject*, PyObject*>>);

  static const impl::SavedTensorDefaultHooksTLS& get_tls_state();
  static void set_tls_state(const impl::SavedTensorDefaultHooksTLS& tls);

  // NOTE: [Disabling SavedTensorDefaultHooks]
  // A developer of a PyTorch feature may choose to disable SavedTensorDefault
  // hooks, especially if their feature does not work with it. If they are
  // disabled, then the following will raise an error:
  // - Attempting to push_hooks
  // - calling disable(message) with a non-zero stack (from get_stack) size
  static void disable(const std::string& error_message);
  static void enable();
  static bool is_enabled();
  static const c10::optional<std::string>& get_disabled_error_message();
};

} // namespace at
