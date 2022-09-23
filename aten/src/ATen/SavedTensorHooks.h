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

  // See NOTE: [Disabling SavedTensorDefaultHooks]
  c10::optional<std::string> disabled_error_message;
  bool is_disabled = false;
};

} // namespace impl

struct TORCH_API SavedTensorDefaultHooks {
  static void push_hooks(PyObject* pack_hook, PyObject* unpack_hook);
  static void pop_hooks();
  static std::pair<PyObject*, PyObject*> get_hooks();
  static void enable(); // NB: has nothing to do with disable() below.
  static std::stack<std::pair<PyObject*, PyObject*>> get_stack();
  static void set_stack(std::stack<std::pair<PyObject*, PyObject*>>);

  static const impl::SavedTensorDefaultHooksTLS& get_tls_state();
  static void set_tls_state(const impl::SavedTensorDefaultHooksTLS& tls);

  // NOTE: [Disabling SavedTensorDefaultHooks]
  // A developer of a PyTorch feature may choose to disable SavedTensorDefault
  // hooks, especially if their feature does not work with it. If they are
  // disabled, then the following will raise an error:
  // - Attempting to push_hooks
  // - calling set_disabled(true) with a non-zero stack (from get_stack) size
  static void set_disabled(bool disabled);
  static bool is_disabled();
  static const optional<std::string>& get_disabled_error_message();
  static void set_disabled_error_message(const optional<std::string>& message);
};

} // namespace at
